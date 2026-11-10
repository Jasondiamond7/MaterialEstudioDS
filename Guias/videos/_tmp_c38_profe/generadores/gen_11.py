import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, alerta, fila, codigo, flujo)

OUT = out("guion_c38_11_roc.json")
v = V("ROC, precisión y el protocolo del spam")
SR = {r["modelo"]: r for r in D["s_ranking"]}
BO = SR["BOSQUE_8"]
PT = D["s_puntos"]
RC = D["s_roc"]
CA = D["s_cal"]
SP = D["s_part"]
C05 = next(c for c in D["s_costos"] if abs(c["t"] - 0.5) < 1e-9)

v.portada("ROC y el protocolo del spam", "Curvas de ordenamiento, calibración y la elección del modelo",
          "Undécima parada: las curvas ROC y de precisión y recall, y el protocolo con el que la clase elige el modelo de spam.")

# ---- 1. Hook
v.b("El bosque aleatorio de la clase tiene un área bajo la curva ROC de 0.975. [[a]]¿Eso significa que acierta el 97.5 por ciento de los correos? [[b]]No. Con el umbral de 0.5, acierta el 93.1 por ciento. [[c]]El área bajo la curva mide otra cosa: qué tan bien ordena los correos, sobre todos los umbrales a la vez. Veamos qué significa.",
    CL(), T("AUC = 0.975: ¿97.5 % de aciertos?", 960, 150, tam=56, c="tinta", ent="subir"),
    N(BO["auc"], 560, 400, "{:.3f}", 110, "violeta", en="a"), T("área bajo la curva ROC", 560, 500, tam=36, c="gris", w="sb", en="a", t=0.3),
    N(100 * C05["acc"], 1360, 400, "{:.1f}", 110, "azul", suf=" %", en="b"), T("aciertos con t = 0.5", 1360, 500, tam=36, c="gris", w="sb", en="b", t=0.3),
    P("la AUC resume el ordenamiento, no los aciertos", 960, 720, "ambar", 48, en="c"), real(1050, en="a"), sec="1. AUC no es exactitud")

# ---- 2. ROC y PR
v.b("La diapositiva cuarenta y cinco presenta dos curvas. [[a]]La curva ROC: en el eje vertical, el recall o sensibilidad, verdaderos positivos sobre todos los spam reales. En el horizontal, la tasa de falsos positivos: falsos positivos sobre todos los legítimos. [[b]]La curva de precisión y recall: en el eje vertical, la precisión, verdaderos positivos sobre todo lo marcado; en el horizontal, el recall. [[c]]En las dos curvas, cada punto corresponde a un umbral.",
    CL(), T("ROC y precision-recall miran aspectos diferentes · diapositiva 45", 960, 140, tam=44, c="tinta", ent="subir"),
    tarjetas([(560, "violeta", "curva ROC", "vertical: recall = VP/(VP+FN) · horizontal: FP/(FP+VN)", "a"),
              (1360, "teal", "curva PR", "vertical: precisión = VP/(VP+FP) · horizontal: recall", "b")], y=440, w=700, h=340, tam_t=48, tam_d=34),
    P("cada punto = un umbral", 960, 760, "ambar", 50, en="c"), diapo(45, en="a"), sec="2. Dos curvas")

# ---- 3. Analogía
v.b("Piensa en un detector de metales con una perilla de sensibilidad. [[a]]Con la perilla baja, casi nunca suena: pocas falsas alarmas, pero se le escapan muchas monedas. [[b]]Con la perilla alta, encuentra casi todas las monedas, pero también suena con cada tapita. [[c]]La curva ROC dibuja todas las posiciones de la perilla a la vez: cuánto encuentras, frente a cuántas falsas alarmas pagas.",
    CL(), T("el detector de metales", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "perilla baja", "pocas falsas alarmas · se escapan monedas", "a"), (1360, "rojo", "perilla alta", "encuentra todo · suena con cada tapita", "b")],
             y=440, w=660, h=300, tam_t=50, tam_d=38),
    P("ROC = todas las posiciones de la perilla", 960, 760, "violeta", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="3. Analogía: el detector")

# ---- 4. La ROC real
p5, p6 = PT["0.50"], PT["0.60"]
v.b("Esta es la curva ROC del bosque, con sus probabilidades fuera de pliegue, recalculada. [[a]]La diagonal punteada es un modelo que ordena al azar. [[b]]La curva del bosque se pega a la esquina superior izquierda: mucho recall con pocos falsos positivos. [[c]]Dos de sus puntos: con umbral 0.5, la tasa de falsos positivos es 0.033 y el recall, 0.875. Con 0.6, 0.020 y 0.825. El área bajo la curva es 0.975, y la precisión promedio, que resume la otra curva, 0.966.",
    CL(), T("la ROC del bosque (probabilidades OOF) · diapositiva 51", 960, 140, tam=46, c="tinta", ent="subir"),
    G(520, 230, 1160, 870, 0, 1, 0, 1, [dict(xs=[0, 1], ys=[0, 1], color="gris", grosor=3, revela=0.0, lapiz=False),
                                        dict(xs=RC["fpr"], ys=RC["tpr"], color="violeta", grosor=6, revela=0.0),
                                        dict(xs=[p6["fpr"], p5["fpr"]], ys=[p6["rec"], p5["rec"]], color="ambar", linea=False, marcadores=11, revela=0.0, lapiz=False)],
      xticks=[0, 0.5, 1], yticks=[0, 0.5, 1.0], yfmt="{:.1f}", xlabel="tasa de falsos positivos", ylabel="recall", id="groc", tam_txt=26),
    U("groc", **{"series.0.revela": 1.0}, en="a", dur=0.8), U("groc", **{"series.1.revela": 1.0}, en="b", dur=1.6),
    U("groc", **{"series.2.revela": 1.0}, en="c", dur=0.6),
    T(f"t = 0.5: FPR {p5['fpr']:.3f}, recall {p5['rec']:.3f}", 1240, 420, tam=32, c="ambar", w="sb", anc="lm", en="c", t=0.4),
    T(f"t = 0.6: FPR {p6['fpr']:.3f}, recall {p6['rec']:.3f}", 1240, 480, tam=32, c="ambar", w="sb", anc="lm", en="c", t=2.5),
    T(f"AUC = {BO['auc']:.3f}   ·   AP = {BO['ap']:.3f}", 1240, 640, tam=40, c="violeta", w="b", anc="lm", en="c", t=4.5),
    real(1050, en="a"), sec="4. La ROC del bosque")
v.b("Y aquí la advertencia de la diapositiva: [[a]]el área bajo la curva y la precisión promedio describen el ranking. [[b]]No fijan un umbral, y no certifican que las probabilidades estén calibradas. [[c]]Además, la precisión promedio depende de la proporción de positivos. Veamos por qué.",
    CL(), P("AUC y AP describen el ranking", 960, 320, "violeta", 50, en="a"),
    P("no fijan un umbral · no certifican calibración", 960, 480, "rojo", 46, en="b"),
    P("la AP depende de la proporción de positivos", 960, 640, "ambar", 46, en="c"), diapo(45, en="a"))

# ---- 5. La frecuencia cambia la precisión
v.b("La diapositiva cuarenta y seis lo muestra con mil correos. [[a]]El mismo filtro, con sensibilidad de 80 por ciento y tasa de falsos positivos de 10 por ciento. [[b]]Si la mitad es spam: 400 verdaderos positivos y 50 falsos positivos. La precisión es 400 sobre 450, 88.9 por ciento. [[c]]Si solo el 5 por ciento es spam: 40 verdaderos positivos, pero 95 falsos positivos, porque los legítimos son muchos. La precisión cae a 40 sobre 135: 29.6 por ciento.",
    CL(), T("la frecuencia de spam cambia la precisión · diapositiva 46", 960, 140, tam=46, c="tinta", ent="subir"),
    P("mismo filtro: sensibilidad 80 % · tasa de FP 10 % · 1000 correos", 960, 260, "gris", 40, en="a"),
    T("50 % spam: VP = 0.8 · 500 = 400 · FP = 0.1 · 500 = 50", 960, 380, tam=38, c="tinta", w="sb", en="b"),
    T("precisión = 400 / 450 =", 960, 460, tam=44, c="tinta", w="sb", anc="rm", en="b", t=1.2), N(88.9, 990, 460, "{:.1f}", 60, "verde", anc="lm", suf=" %", en="b", t=1.8),
    T("5 % spam: VP = 0.8 · 50 = 40 · FP = 0.1 · 950 = 95", 960, 600, tam=38, c="tinta", w="sb", en="c"),
    T("precisión = 40 / 135 =", 960, 680, tam=44, c="tinta", w="sb", anc="rm", en="c", t=1.5), N(29.6, 990, 680, "{:.1f}", 60, "rojo", anc="lm", suf=" %", en="c", t=2.1),
    T("una precisión medida en un corpus no se traslada a otro", 960, 820, tam=38, c="gris", w="sb", en="c", t=3.0), diapo(46, en="a"), sec="5. La frecuencia importa")

# ---- 6. El protocolo
v.b("Con estas herramientas, la clase fija un protocolo antes de mirar resultados, en la diapositiva cuarenta y siete. [[a]]Uno: reservar 921 correos para la prueba, y usar 3680 para ajustar y seleccionar. [[b]]Dos: comparar tres candidatos por log loss fuera de pliegue, con cinco pliegues. Tres: en el ganador, elegir el umbral, entre 0.05 y 0.95, por el menor costo fuera de pliegue. [[c]]Cuatro: reajustar y evaluar una sola vez el modelo y el umbral elegidos. Primero, la calidad de las probabilidades; después, la decisión.",
    CL(), T("un protocolo para probabilidades y decisiones · diapositiva 47", 960, 140, tam=46, c="tinta", ent="subir"),
    flujo([("1 · reservar\n921 de prueba", "gris", "a"), ("2 · candidatos\npor log loss OOF", "violeta", "b"), ("3 · umbral\npor costo OOF", "ambar", "b"),
           ("4 · reajustar\ny evaluar una vez", "teal", "c")], y=440, w=360, h=230, x0=300, paso=440, tam=34),
    P("primero calidad probabilística, después decisión", 960, 680, "violeta", 44, en="c", t=1.5),
    T("no busca el menor costo entre TODAS las parejas modelo-umbral · empates: orden de modelos, mayor umbral", 960, 800, tam=30, c="gris", w="sb",
      en="c", t=2.5), diapo(47, en="a"), sec="6. El protocolo")

# ---- 7. Candidatos
v.b("Los tres candidatos, en la diapositiva cuarenta y ocho, son instrumentos de comparación. [[a]]La frecuencia: una probabilidad constante, igual a la proporción de spam en las filas de ajuste, cerca de 0.394. Es la referencia sin predictores. [[b]]La regresión logística, con la escala aprendida dentro del ajuste y penalización fija, C igual a 1. [[c]]Y el bosque aleatorio: el promedio de 150 árboles, con profundidad máxima 8 y al menos cinco casos por hoja. En R, el bosque se ajusta con ranger; en Python, con scikit-learn.",
    CL(), T("los candidatos · diapositiva 48", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "frecuencia", "p constante = proporción de spam (≈ 0.394)", "a"),
              (960, "azul", "logística", "escala dentro del ajuste · C = 1", "b"),
              (1520, "verde", "bosque", "150 árboles · profundidad 8 · ≥ 5 por hoja", "c")], y=460, w=500, h=340, tam_t=46, tam_d=34),
    T("R: ranger   ·   Python: scikit-learn", 960, 760, tam=38, c="gris", w="sb", en="c", t=1.5), diapo(48, en="a"), sec="7. Los candidatos")

# ---- 8. Resultados OOF
v.b("Y los resultados fuera de pliegue, en los 3680 correos, como en la diapositiva cincuenta. [[a]]La frecuencia tiene un log loss de 0.671, y un Brier de 0.239. [[b]]La logística: 0.275 y 0.075. [[c]]Y el bosque: 0.228 y 0.061. El bosque obtiene el menor log loss, y la regla fijada lo elige.",
    CL(), T("la regla fijada elige por log loss OOF · diapositiva 50", 960, 140, tam=46, c="tinta", ent="subir"),
    fila(290, [("", 560), ("log loss", 1060), ("Brier", 1400)], tams=[40] * 3, colores=["gris"] * 3),
    fila(400, [("frecuencia", 560), (f"{SR['FRECUENCIA']['log_loss']:.3f}", 1060), (f"{SR['FRECUENCIA']['brier']:.3f}", 1400)],
         tams=[42, 54, 54], colores=["gris"] * 3, en="a", fondo="linea", x0=360, x1=1580, h=100),
    fila(520, [("logística (C = 1)", 560), (f"{SR['LOGISTICA_C1']['log_loss']:.3f}", 1060), (f"{SR['LOGISTICA_C1']['brier']:.3f}", 1400)],
         tams=[42, 54, 54], colores=["azul"] * 3, en="b", fondo="azulclaro", x0=360, x1=1580, h=100),
    fila(640, [("bosque (prof. 8)", 560), (f"{BO['log_loss']:.3f}", 1060), (f"{BO['brier']:.3f}", 1400)],
         tams=[42, 54, 54], colores=["verde"] * 3, en="c", fondo="verdeclaro", x0=360, x1=1580, h=100),
    P("menor log loss → se elige el bosque", 960, 800, "verde", 48, en="c", t=1.2), real(1050, en="a"), sec="8. Los resultados")
v.b("Un cálculo para entender la referencia. [[a]]La frecuencia dice, en cada pliegue, cerca de 0.394. Su log loss es: 0.394 por menos el logaritmo de 0.394, más 0.606 por menos el logaritmo de 0.606. [[b]]Eso es 0.394 por 0.931, más 0.606 por 0.501: 0.367 más 0.304, igual a 0.671. [[c]]Y su Brier es simplemente 0.394 por 0.606: 0.239. Son el piso de no usar ningún predictor.",
    CL(), T("la referencia, a mano", 960, 150, tam=54, c="tinta", ent="subir"),
    T("log loss = 0.394 · (−ln 0.394) + 0.606 · (−ln 0.606)", 960, 300, tam=42, c="tinta", w="sb", en="a"),
    T("= 0.394 · 0.931 + 0.606 · 0.501 = 0.367 + 0.304 =", 960, 400, tam=42, c="tinta", w="sb", anc="mm", en="b"),
    N(0.671, 960, 500, "{:.3f}", 70, "gris", en="b", t=1.8),
    T("Brier = 0.394 · 0.606 =", 1000, 640, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(0.239, 1030, 640, "{:.3f}", 64, "gris", anc="lm", en="c", t=0.8),
    P("el piso de no usar predictores", 960, 790, "teal", 46, en="c", t=1.6), adicional(1050, en="a"))

# ---- 9. Alerta del material
v.b("Aquí conviene corregir algo de la grabación. [[a]]Al presentar este protocolo se dijo: «voy a ocupar la pérdida cuadrática fuera de pliegue, y voy a escoger el ganador». [[b]]La pérdida cuadrática de una probabilidad es el Brier. Pero el script elige por log loss: ordena el ranking por la columna log loss. [[c]]Y la diapositiva cincuenta lo dice explícito: el log loss selecciona; el Brier describe. En estos datos los dos eligen el bosque, pero el criterio fijado es el log loss.",
    CL(), alerta(170), cita("voy a ocupar la pérdida cuadrática fuera de pliegue, y voy a escoger el ganador", 320, en="a", tam=42),
    etiqueta("grabación de la clase", 410, en="a", t=0.4),
    codigo(["ranking = pd.DataFrame(rows).sort_values(['log_loss', 'orden'])"], y=560, w=1500, tam=36, en="b"),
    P("diapositiva 50: el log loss selecciona; el Brier describe", 960, 720, "verde", 44, en="c"),
    T("aquí ambos eligen el bosque, pero el criterio fijado es el log loss", 960, 840, tam=36, c="gris", w="sb", en="c", t=1.5), sec="9. Una corrección")

# ---- 10. Calibración del ganador
CX = [c["p"] for c in CA]
CY = [c["f"] for c in CA]
v.b("Elegido el bosque, falta revisar su calibración: diapositiva cincuenta y dos. [[a]]Sus probabilidades fuera de pliegue se ordenan y se dividen en cinco grupos de 736 correos. Para cada grupo, la probabilidad media en el eje horizontal, y la fracción real de spam en el vertical. [[b]]Bajo la diagonal, el modelo sobreestima; sobre la diagonal, subestima. [[c]]En el cuarto grupo: predice 0.673 y se observa 0.814. Un buen ranking no garantiza probabilidades calibradas.",
    CL(), T("una AUC alta no garantiza calibración · diapositiva 52", 960, 140, tam=46, c="tinta", ent="subir"),
    G(520, 230, 1160, 870, 0, 1, 0, 1, [dict(xs=[0, 1], ys=[0, 1], color="gris", grosor=3, revela=0.0, lapiz=False),
                                        dict(xs=CX, ys=CY, color="azul", grosor=5, marcadores=11, revela=0.0)],
      xticks=[0, 0.5, 1], yticks=[0, 0.5, 1.0], yfmt="{:.1f}", xlabel="probabilidad media OOF", ylabel="spam", id="gcal", tam_txt=26),
    U("gcal", **{"series.0.revela": 1.0}, en="a", t=1.0, dur=0.8), U("gcal", **{"series.1.revela": 1.0}, en="a", t=2.0, dur=1.6),
    T("sobre la diagonal: subestima", 1240, 380, tam=34, c="tinta", w="sb", anc="lm", en="b"), T("bajo la diagonal: sobreestima", 1240, 440, tam=34, c="tinta", w="sb", anc="lm", en="b", t=0.8),
    R(520 + CA[3]["p"] * 640, 870 - CA[3]["f"] * 640, 70, 70, "rojo", r=35, en="c"),
    T(f"4.º grupo: predice {CA[3]['p']:.3f}, observa {CA[3]['f']:.3f}", 1240, 580, tam=34, c="rojo", w="sb", anc="lm", en="c", t=0.4),
    real(1050, en="a"), sec="10. La calibración del ganador")
v.b("¿Y qué hacemos con eso? [[a]]Se podría recalibrar el modelo. Pero ojo: recalibrar cambia la receta, y la nueva receta exige su propia validación. [[b]]Por ahora, el protocolo sigue con el bosque tal como está. [[c]]Y esta subestimación va a reaparecer en la próxima parada, cuando comparemos el umbral teórico con el umbral elegido.",
    CL(), P("recalibrar = cambiar la receta → nueva validación", 960, 320, "ambar", 46, en="a"),
    P("el protocolo sigue con el bosque tal cual", 960, 480, "verde", 46, en="b"),
    T("la subestimación reaparece al elegir el umbral", 960, 640, tam=42, c="tinta", w="sb", en="c"), diapo(52, en="a"))

# ---- 11. El error típico
v.b("El error típico: [[a]]leer el área bajo la curva como un porcentaje de aciertos. [[b]]Un 0.975 se lee mejor así: si tomo al azar un spam y un legítimo, en el 97.5 por ciento de los casos el spam recibe una probabilidad más alta. [[c]]Es una medida de orden, no de aciertos, y no dice qué umbral usar.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("AUC = porcentaje de aciertos", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("AUC 0.975: un spam y un legítimo al azar →", 960, 440, tam=42, c="tinta", w="sb", en="b"),
    T("el spam recibe más probabilidad en el 97.5 % de los pares", 960, 520, tam=42, c="tinta", w="sb", en="b", t=1.0),
    P("medida de orden, no de aciertos", 960, 700, "violeta", 50, en="c"), sec="11. El error típico")

# ---- 12. Piensa un segundo
v.b("Piensa un segundo. [[a]]Mismo filtro: sensibilidad de 80 por ciento y tasa de falsos positivos de 10 por ciento. Pero ahora, en mil correos, el 20 por ciento es spam. [[b]]¿Cuál es la precisión?",
    CL(), T("sensibilidad 80 % · tasa de FP 10 %", 960, 280, tam=54, c="tinta", w="b", en="a"),
    T("1000 correos · 20 % spam", 960, 380, tam=50, c="ambar", w="sb", en="a", t=1.0),
    P("precisión = ?", 960, 560, "violeta", 56, en="b"), sec="12. Piensa un segundo", pausa=0.6)
v.b("Hay 200 spam y 800 legítimos. [[a]]Verdaderos positivos: el 80 por ciento de 200, 160. [[b]]Falsos positivos: el 10 por ciento de 800, 80. [[c]]Precisión: 160 sobre 240, 66.7 por ciento. Entre el 88.9 y el 29.6 de la diapositiva, como era de esperar.",
    CL(), T("VP = 0.8 · 200 =", 1000, 300, tam=52, c="tinta", w="sb", anc="rm", en="a"), N(160, 1030, 300, "{:.0f}", 64, "verde", anc="lm", en="a", t=0.6),
    T("FP = 0.1 · 800 =", 1000, 420, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(80, 1030, 420, "{:.0f}", 64, "rojo", anc="lm", en="b", t=0.6),
    T("precisión = 160 / 240 =", 1000, 560, tam=52, c="tinta", w="sb", anc="rm", en="c"), N(66.7, 1030, 560, "{:.1f}", 64, "violeta", anc="lm", suf=" %", en="c", t=0.8),
    ilustrativo(1000))

# ---- 13. Para llevar
v.b("Para llevarte: [[a]]la ROC y la curva de precisión y recall describen el ordenamiento sobre todos los umbrales; la precisión depende de la frecuencia de positivos. [[b]]El protocolo elige primero el modelo con la pérdida logarítmica fuera de pliegue, y después el umbral por costo. [[c]]Y un buen ranking no garantiza buena calibración.",
    CL(), P("ROC y PR: el orden · la precisión depende de la frecuencia", 960, 300, "violeta", 42, en="a"),
    P("modelo por log loss OOF → umbral por costo OOF", 960, 460, "verde", 46, en="b"),
    P("buen ranking ≠ buena calibración", 960, 620, "ambar", 48, en="c"), sec="13. Para llevar")
v.b("Siguiente parada, la última: [[a]]elegir el umbral por costo, el umbral teórico, la evaluación final en la prueba, y el repaso de toda la clase.",
    CL(), siguiente(12, "Umbral por costo, prueba final y repaso"), sec="14. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
