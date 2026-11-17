import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_12_repaso.json")
v = V("Límites, bibliotecas y repaso")
PR, I1 = D["prueba"], D["id1"]

v.portada("Límites, bibliotecas y repaso", "Qué revisar antes de usar el filtro, y la clase completa en un vistazo",
          "Última parada. Vamos a ver qué hay que revisar antes de llevar este filtro a otra bandeja de correo, qué significa de verdad una probabilidad alta, cómo se hace todo con bibliotecas, y el repaso de la clase.")

# ---- 1. Otra bandeja
v.b("La diapositiva cincuenta y cinco conecta lo aprendido con lo que hay que revisar en una bandeja nueva. [[a]]Primero, la frecuencia de spam: con oferta, pasar de veinte a dos por ciento de spam llevó la probabilidad de 66.7 a 14.0 por ciento. Hay que revisar si la frecuencia de spam del ajuste representa a los nuevos correos. [[b]]Segundo, copiar una señal elevó la certeza sin agregar información: hay que revisar la dependencia entre señales y la calibración. [[c]]Y tercero, Spambase contiene correos históricos: falta ver resultados en correos actuales, aún no utilizados.",
    CL(), T("qué revisar antes de usar el filtro en otra bandeja · diapositiva 55", 960, 130, tam=44, c="tinta", ent="subir"),
    tabla([["lo que aprendimos", "qué revisar en la nueva bandeja"], ["con oferta: 66.7 % → 14.0 % (spam de 20 % a 2 %)", "¿la frecuencia de spam sigue representando?"],
           ["copiar una señal elevó la certeza", "dependencia entre señales y calibración"], ["Spambase contiene correos históricos", "resultados en correos actuales"]],
          [530, 1400], 270, paso=120, tam=34, x0=110, x1=1810, ens=[None, "a", "b", "c"]),
    P("cambiar o calibrar el filtro exige evaluar la receta completa con datos separados", 960, 820, "violeta", 38, en="c", t=1.5),
    diapo(55, en="a"), sec="1. Otra bandeja")

# ---- 2. Desbalance
v.b("En la clase hubo una pregunta relacionada: ¿hay que corregir algo si las clases están desbalanceadas? [[a]]El profesor explicó que una opción es rebalancear, por ejemplo llevar los datos a cincuenta y cincuenta; pero que después hay que compensar ese balanceo, porque si no se sobreestiman las probabilidades de la clase rara. [[b]]Con lo que vimos en la parada dos, se entiende por qué. Un cálculo ilustrativo: con la palabra oferta y previa veinte por ciento, la probabilidad es 0.6667. Si el ajuste se hizo con datos rebalanceados al cincuenta por ciento, el modelo diría 0.8889. [[c]]Para volver a la realidad, hay que reponer la previa real.",
    CL(), cita("después tengo que compensar ese balanceo que yo hice ya porque si no voy a sobreestimar cosas", y=260, tam=40),
    tarjetas([(620, "verde", "previa real 20 %", "P(spam | oferta) = 0.6667", "b"), (1300, "rojo", "rebalanceado 50 %", "P(spam | oferta) = 0.8889", "b")],
             y=530, w=560, h=260, tam_t=46, tam_d=42),
    P("después de rebalancear, hay que reponer la previa real", 960, 780, "violeta", 44, en="c"), grabacion("02:00:33"), ilustrativo(1010, en="b"),
    sec="2. Clases desbalanceadas")

# ---- 3. La regla completa
v.b("La diapositiva cincuenta y seis resume la regla completa, de los datos a la acción. [[a]]Representar: cincuenta y cuatro frecuencias convertidas en presencia o ausencia. [[b]]Aprender: frecuencias por clase con suavizado, alfa igual a 0.5. Predecir: combinar presencias y ausencias, y normalizar entre las dos clases. [[c]]Actuar: bloquear cuando la probabilidad de spam sea al menos 0.95. Y comprobar: contar errores y costo, y revisar si las probabilidades reflejan frecuencias.",
    CL(), T("la regla que construimos, de los datos a la acción · diapositiva 56", 960, 120, tam=44, c="tinta", ent="subir"),
    tabla([["paso", "decisión del caso de correos"], ["representar", "54 frecuencias convertidas en presencia o ausencia"], ["aprender", "frecuencias por clase con suavizado α = 0.5"],
           ["predecir", "combinar presencias y ausencias; normalizar entre las dos clases"], ["actuar", "bloquear cuando la probabilidad de spam sea al menos 0.95"],
           ["comprobar", "contar errores y costo; revisar si las probabilidades reflejan frecuencias"]],
          [330, 1120], 240, paso=105, tam=36, x0=140, x1=1790,
          colores=[["tinta"] * 2, ["teal", "tinta"], ["azul", "tinta"], ["violeta", "tinta"], ["ambar", "tinta"], ["verde", "tinta"]], ens=[None, "a", "b", "b", "c", "c"]),
    diapo(56, en="a"), sec="3. La regla completa")
v.b("Y la frase final de esa diapositiva, que conviene memorizar: [[a]]el suavizado y el umbral cumplen tareas distintas. [[b]]El suavizado modifica el cálculo de la probabilidad. El umbral modifica la acción.",
    CL(), P("suavizado → modifica el cálculo de la probabilidad", 960, 360, "teal", 50, en="a"),
    P("umbral → modifica la acción", 960, 540, "ambar", 54, en="b"), diapo(56, en="a"))

# ---- 4. Una probabilidad muy alta
v.b("La diapositiva cincuenta y ocho vuelve al correo explicado, con probabilidad 0.9964, y pregunta qué podemos concluir. [[a]]Sí: bajo ese ajuste de Bernoulli, las señales del correo favorecen mucho a spam frente a legítimo. [[b]]Aún no: que el 99.64 por ciento de los correos parecidos sea realmente spam. Esa lectura requiere comprobar la calibración, y en la parada diez vimos que el modelo exagera. [[c]]Tampoco: que el correo sea habitual. Un clasificador reparte la probabilidad entre sus clases aunque el caso resulte extraño.",
    CL(), T("¿qué significa una probabilidad de spam muy alta? · diapositiva 58", 960, 130, tam=44, c="tinta", ent="subir"),
    K(300, 300, "check", 64, en="a"), T("sí: las señales favorecen mucho a spam frente a legítimo", 380, 300, tam=40, c="verde", anc="lm", en="a"),
    K(300, 450, "cruz", 64, en="b"), T("aún no: «99.64 % de los parecidos son spam» (falta calibración)", 380, 450, tam=40, c="rojo", anc="lm", en="b"),
    K(300, 600, "cruz", 64, en="c"), T("tampoco: que el correo sea habitual", 380, 600, tam=40, c="rojo", anc="lm", en="c"),
    P("una probabilidad alta expresa una preferencia del modelo", 960, 790, "violeta", 44, en="c", t=1.0), diapo(58, en="a"), sec="4. Una probabilidad muy alta")

# ---- 5. Analogía
v.b("Una analogía para ese último punto. [[a]]Si a un detector de spam le muestras una receta de cocina en otro idioma, igual tiene que repartir la probabilidad entre spam y legítimo: las dos suman uno. [[b]]Puede decir noventa y cinco por ciento legítimo, aunque el correo no se parezca a nada que haya visto. [[c]]La probabilidad de clase no mide qué tan raro es el caso: esa es otra pregunta.",
    CL(), T("un correo que no se parece a nada", 960, 150, tam=56, c="tinta", ent="subir"),
    P("spam + legítimo = 1, siempre", 960, 340, "teal", 52, en="a"),
    P("puede decir «95 % legítimo» ante algo nunca visto", 960, 500, "ambar", 48, en="b"),
    P("probabilidad de clase ≠ qué tan raro es el caso", 960, 660, "violeta", 48, en="c"), analogia(1050), sec="5. Analogía: lo nunca visto")

# ---- 6. Bibliotecas
v.b("Por último, las bibliotecas, de la diapositiva sesenta y uno. Automatizan el ajuste y la predicción que construimos paso a paso. [[a]]En R, el paquete naivebayes tiene una función para cada variante: Bernoulli, gaussiana y multinomial. [[b]]En Python, scikit-learn tiene una clase para cada una, con los mismos nombres de variante. [[c]]Para obtener probabilidades: en R, predict con el tipo prob; en Python, el método predict proba.",
    CL(), T("Naive Bayes en R y Python · diapositiva 61", 960, 120, tam=50, c="tinta", ent="subir"),
    tabla([["variante", "R: naivebayes", "Python: scikit-learn"], ["Bernoulli", "bernoulli_naive_bayes()", "BernoulliNB()"],
           ["gaussiana", "gaussian_naive_bayes()", "GaussianNB()"], ["multinomial", "multinomial_naive_bayes()", "MultinomialNB()"],
           ["probabilidades", 'predict(modelo, nuevos, type="prob")', "modelo.predict_proba(nuevos)"]],
          [300, 880, 1530], 250, paso=105, tam=34, x0=110, x1=1810, ens=[None, "a", "a", "a", "c"]),
    P("Bernoulli usa entradas binarias · suavizado: laplace en R, alpha en Python", 960, 820, "ambar", 40, en="b", t=1.0),
    diapo(61, en="a"), sec="6. Bibliotecas")
v.b("En el script de la clase, el Bernoulli se ajusta exactamente así. [[a]]El argumento de binarización, igual a cero, convierte cada frecuencia positiva en presencia: es la representación que elegimos. [[b]]Y la diapositiva cierra con una idea clave: la biblioteca automatiza las cuentas, pero la representación, la validación y el umbral los seguimos decidiendo nosotros.",
    CL(), codigo(["m = BernoulliNB(alpha=0.5, binarize=0.0)", "m.fit(X, y)", "p = m.predict_proba(X_nuevo)[:, 1]"], y=330, w=1300, tam=40, c="violeta", en="a"),
    P("binarize = 0.0: frecuencia > 0 → presencia", 960, 580, "teal", 46, en="a", t=1.0),
    P("la biblioteca automatiza las cuentas; nosotros decidimos representación, validación y umbral", 960, 760, "ambar", 38, en="b"),
    real(1050, en="a", texto="Mismo ajuste del script de la clase (tramo 05)"))

# ---- 7. Síntesis de la clase
v.b("La diapositiva sesenta cierra con la pregunta del inicio: ¿basta con que un correo contenga oferta para bloquearlo? [[a]]Uno: construir la probabilidad. En los cien correos, oferta sin enlace dio 0.4140 con suavizado: también importan las ausencias y la frecuencia de clase. [[b]]Dos: comprobar su calidad. En Spambase se eligió Bernoulli con alfa 0.5 por el menor log loss fuera de pliegue. [[c]]Tres: decidir con un propósito. Con costos de cinco y uno, se eligió el umbral 0.95 en desarrollo.",
    CL(), T("de observar una palabra a decidir qué hacer · diapositiva 60", 960, 130, tam=46, c="tinta", ent="subir"),
    flujo([("construir: 0.4140 con suavizado", "violeta", "a"), ("comprobar: Bernoulli α = 0.5 (log loss OOF)", "teal", "b"),
           ("decidir: umbral 0.95 (costos 5 y 1)", "ambar", "c")], y=440, w=500, h=260, x0=400, paso=560, tam=38),
    T("una señal aporta información; Bayes la combina; la evaluación contrasta la probabilidad; el costo orienta la decisión", 960, 720, tam=36, c="tinta", w="sb", mw=1600, en="c", t=1.5),
    diapo(60, en="a"), sec="7. Síntesis")

# ---- 8. Repaso de las paradas
v.b("Repasemos el recorrido con sus números. [[a]]Previa, condicional y posterior: oferta dado spam 0.80, spam dado oferta dos tercios. Con dos por ciento de spam, 0.1404. [[b]]Independencia dentro de cada clase: oferta sin enlace, 0.4286; con Laplace, 0.4140. [[c]]Variantes: la representación decide; con porcentajes, Bernoulli frente a la gaussiana sobre el logaritmo de uno más equis.",
    CL(), T("repaso: la primera mitad", 960, 140, tam=54, c="tinta", ent="subir"),
    P("P(oferta | spam) = 0.80 · P(spam | oferta) = 2/3 · con 2 % de spam: 0.1404", 960, 300, "violeta", 40, en="a"),
    P("independencia: 0.4286 · con Laplace (α = 1): 0.4140", 960, 460, "teal", 44, en="b"),
    P("la representación decide la variante", 960, 620, "azul", 46, en="c"), sec="8. Repaso")
v.b("Y la segunda mitad. [[a]]Fuera de pliegue gana Bernoulli con alfa 0.5, log loss 0.556; la gaussiana acierta 80.8 por ciento con log loss 4.931. [[b]]Las señales dependientes exageran la certeza: en el intervalo de arriba, 99.44 por ciento predicho frente a 92.13 observado. [[c]]Y la decisión: umbral 0.95, con costo 219.0 fuera de pliegue y 188.9 en la prueba histórica, frente a 394.1 sin filtrar.",
    CL(), T("repaso: la segunda mitad", 960, 140, tam=54, c="tinta", ent="subir"),
    P("OOF: Bernoulli α = 0.5 (0.556) · gaussiana 80.8 % pero 4.931", 960, 300, "verde", 42, en="a"),
    P("calibración: 99.44 % predicho frente a 92.13 % observado", 960, 460, "ambar", 44, en="b"),
    P("umbral 0.95: 219.0 (OOF) · 188.9 (prueba) frente a 394.1", 960, 620, "violeta", 44, en="c"))

# ---- 9. Precisiones de la grabación
v.b("En la serie marcamos algunas precisiones sobre la grabación; aquí van juntas. [[a]]Catorce spam tienen enlace, y seis no. Los seis sin enlace son spam, no legítimos. Y los veinte spam son veinte de cien. [[b]]El suavizado cambió la probabilidad en cerca de un punto y medio. El correo de identificador uno tiene 0.9964. Y las frecuencias del archivo son porcentajes, no van de cero a uno. [[c]]La comparación de recetas usa predicciones fuera de pliegue, no la prueba. El ejemplo de las copias tiene cien spam. Y copiar una señal no es sobreajuste, sino evidencia contada dos veces.",
    CL(), T("precisiones sobre la grabación, en un vistazo", 960, 110, tam=50, c="tinta", ent="subir"),
    tabla([["minuto", "lo correcto"], ["00:32:22", "14 spam con enlace, 6 sin enlace"], ["00:49:54", "los 6 sin enlace son spam (no legítimos)"],
           ["01:07:22", "los spam son 20 de los 100"], ["01:14:58", "0.4286 → 0.4140: cerca de 1.5 puntos"], ["01:32:34", "identificador 1: p ≈ 0.9964"],
           ["01:36:27", "frecuencias en porcentaje (máximo 42.81)"], ["01:48:57", "las recetas se comparan con OOF, no con la prueba"],
           ["01:52:16", "100 spam y 100 legítimos en el experimento"], ["02:01:47", "copias: evidencia contada dos veces, no sobreajuste"]],
          [380, 1120], 200, paso=82, tam=32, x0=180, x1=1740, ens=[None, "a", "a", "a", "b", "b", "b", "c", "c", "c"]),
    sec="9. Precisiones")

# ---- 10. Criterio de dominio
v.b("Para comprobar que dominas la clase, intenta responder sin mirar. [[a]]¿Puedes calcular a mano la probabilidad de spam desde una tabla de frecuencias, distinguiendo previa, condicional y posterior? [[b]]¿Puedes explicar qué supone el modelo, y qué pasa si dos señales repiten la misma información? ¿Y aplicar el suavizado a un conteo cero? [[c]]¿Puedes explicar por qué se elige con log loss y no con exactitud, y elegir un umbral con costos, probándolo una sola vez?",
    CL(), T("criterio de dominio", 960, 140, tam=56, c="tinta", ent="subir"),
    K(300, 300, "check", 56, en="a"), T("calcular P(spam | señales) a mano: previa, condicional, posterior", 370, 300, tam=38, c="tinta", w="sb", anc="lm", en="a"),
    K(300, 420, "check", 56, en="b"), T("explicar el supuesto y el efecto de señales repetidas", 370, 420, tam=38, c="tinta", w="sb", anc="lm", en="b"),
    K(300, 540, "check", 56, en="b"), T("aplicar el suavizado de Laplace a un conteo cero", 370, 540, tam=38, c="tinta", w="sb", anc="lm", en="b"),
    K(300, 660, "check", 56, en="c"), T("elegir con log loss y fijar el umbral por costo; probar una vez", 370, 660, tam=38, c="tinta", w="sb", anc="lm", en="c"),
    sec="10. Criterio de dominio")

# ---- 11. Cierre
v.b("Y con eso cerramos la clase cuarenta. [[a]]La idea que te llevas: un clasificador probabilístico entrega una probabilidad, no solo una etiqueta, y predecir no es decidir. [[b]]El clasificador bayesiano ingenuo combina señales con un supuesto simple, se evalúa con predicciones fuera de pliegue, y se convierte en acción con un umbral que sale de los costos. Nos vemos en la próxima clase.",
    CL(), T("predecir no es decidir", 960, 330, tam=80, c="violeta", w="b", ent="subir", en="a"),
    P("Bayes combina · la evaluación contrasta · el costo decide", 960, 540, "teal", 48, en="b"),
    T("Clase 40 · Naive Bayes y clasificación probabilística", 960, 760, tam=38, c="gris", w="sb", en="b", t=1.0), sec="11. Cierre")

n = v.guardar(OUT)
print(n, "beats")
