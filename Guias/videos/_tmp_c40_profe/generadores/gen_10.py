import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_10_calibracion.json")
v = V("Señales copiadas y calibración")
DU, DR, C9, I1 = D["dup"], D["dup_real"], D["cal9"], D["id1"]
CB = [c for c in D["cal"] if c["modelo"] == "BERN_1"]
CG = [c for c in D["cal"] if c["modelo"] == "GAUSS_LOG"]

v.portada("Señales copiadas y calibración", "Por qué Naive Bayes sale demasiado seguro, y cómo leer un correo real",
          "Décima parada. Vamos a ver qué pasa cuando el supuesto ingenuo falla de la forma más extrema posible: copiando una señal. Después miramos la calibración, y explicamos el correo de identificador uno señal por señal.")

# ---- 1. Experimento controlado
v.b("La diapositiva cuarenta y seis propone un experimento controlado. [[a]]Cien correos spam y cien legítimos. Una señal aparece en ochenta spam y en veinte legítimos. [[b]]La probabilidad de spam al ver la señal: 0.5 por 0.8, dividido por 0.5 por 0.8 más 0.5 por 0.2. [[c]]Exactamente 0.8000. Hasta aquí, nada nuevo.",
    CL(), T("copiar una señal aumenta la certeza · diapositiva 46", 960, 140, tam=48, c="tinta", ent="subir"),
    P("100 spam y 100 legítimos  ·  la señal: en 80 spam y en 20 legítimos", 960, 290, "teal", 42, en="a"),
    T("P(spam | señal) = 0.5 × 0.8 / (0.5 × 0.8 + 0.5 × 0.2) =", 1180, 460, tam=46, c="tinta", w="sb", anc="rm", en="b"),
    N(DU["post"][0], 1210, 460, "{:.4f}", 66, "violeta", anc="lm", en="c"), diapo(46, en="a"), sec="1. Copiar una señal")
v.b("Ahora copiamos esa misma columna, con otro nombre. [[a]]Para Naive Bayes son dos señales independientes, así que multiplica las dos copias: 0.5 por 0.8 por 0.8, sobre lo mismo más 0.5 por 0.2 por 0.2. [[b]]Resultado: aproximadamente 0.9412. [[c]]El correo es el mismo, y la información es la misma. El modelo contó dos veces una señal, y aumentó su certeza. El profesor lo contrastó con la regresión: allí, dos predictores idénticos no suman información.",
    T("con dos copias:  0.5 × 0.8 × 0.8 / (0.5 × 0.8 × 0.8 + 0.5 × 0.2 × 0.2) ≈", 1300, 620, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(DU["post"][1], 1330, 620, "{:.4f}", 64, "rojo", anc="lm", en="b"),
    P("misma información, contada dos veces → más certeza", 960, 790, "ambar", 46, en="c"))

# ---- 2. Alerta
v.b("Una precisión sobre la grabación. [[a]]Al plantear este ejemplo, cerca del minuto ciento doce, se dijo que de los doscientos correos, ochenta eran spam. [[b]]En el ejemplo hay cien spam y cien legítimos; el ochenta es el número de spam que traen la señal.",
    CL(), error_material("01:52:16", "tengo 200 correos en mi conjunto de datos, de los cuales 80 son spam",
                         "100 spam y 100 legítimos; la señal aparece en 80 spam y en 20 legítimos", en_ok="b"),
    sec="2. Una precisión")

# ---- 3. Cuatro copias
GD = "gd10"
v.b("La diapositiva cuarenta y siete sigue copiando. [[a]]Con cuatro copias, los factores se repiten cuatro veces: 0.5 por 0.8 a la cuarta, sobre lo mismo más 0.5 por 0.2 a la cuarta. [[b]]La probabilidad sube a aproximadamente 0.9961. [[c]]La línea horizontal marca la información que realmente hay: una sola señal, 0.8. La clase predicha no cambia; lo que crece es la certeza.",
    CL(), T("la clase se mantiene; la certeza sigue creciendo · diapositiva 47", 960, 130, tam=46, c="tinta", ent="subir"),
    G(330, 240, 1180, 820, 0.5, 4.5, 0.65, 1.02, [dict(xs=DU["copias"], ys=DU["post"], color="ambar", grosor=6, marcadores=12, revela=0.0),
                                                  dict(xs=[0.5, 4.5], ys=[0.8, 0.8], color="azul", grosor=4, revela=0.0)],
      xticks=[1, 2, 4], yticks=[0.7, 0.8, 0.9, 1.0], yfmt="{:.1f}", xlabel="número de copias de la señal", id=GD, tam_txt=28), T("probabilidad de clase 1", 330, 200, tam=30, c="tinta", w="sb", anc="lm"),
    U(GD, **{"series.0.revela": 1.0}, en="a", dur=1.8), U(GD, **{"series.1.revela": 1.0}, en="c", dur=1.0),
    T("1 copia: 0.8000", 1520, 340, tam=40, c="ambar", w="b"), T("2 copias: 0.9412", 1520, 420, tam=40, c="ambar", w="b", en="a"),
    T("4 copias: 0.9961", 1520, 500, tam=40, c="ambar", w="b", en="b"),
    P("información disponible: 0.8", 1520, 640, "azul", 38, en="c"), diapo(47, en="a"), sec="3. Cuatro copias")
v.b("Hay una forma muy limpia de verlo, un cálculo adicional con razones. [[a]]Con una señal, la razón entre spam y legítimo es ochenta a veinte: cuatro a uno. [[b]]Cada copia vuelve a multiplicar por cuatro: dieciséis a uno con dos copias, doscientos cincuenta y seis a uno con cuatro. [[c]]Por eso la certeza crece tan rápido: en logaritmos, cada copia suma la misma evidencia otra vez.",
    CL(), T("cada copia multiplica la razón por 4", 960, 150, tam=54, c="tinta", ent="subir"),
    tabla([["copias", "razón spam : legítimo", "probabilidad"], ["1", "4 : 1", "4/5 = 0.8000"], ["2", "16 : 1", "16/17 ≈ 0.9412"],
           ["4", "256 : 1", "256/257 ≈ 0.9961"]], [560, 980, 1400], 300, paso=110, tam=44, x0=300, x1=1660, ens=[None, "a", "b", "b"]),
    P("en logaritmos: cada copia suma la misma evidencia otra vez", 960, 790, "violeta", 44, en="c"), adicional(1050, en="a"))

# ---- 5. Alerta sobreajuste
v.b("Aquí hubo una pregunta en clase: ¿esto es sobreajuste? [[a]]La respuesta en la grabación fue que sí, que es un sobreajuste porque se cuenta dos veces lo mismo. [[b]]Conviene precisarlo. Sobreajuste es aprender el ruido de los datos de entrenamiento. Aquí ocurre otra cosa: la misma evidencia se cuenta dos veces porque falla el supuesto de independencia, y el resultado es exceso de confianza. [[c]]Y tampoco es una mejora: la probabilidad sube, pero no se agregó información.",
    CL(), error_material("02:01:47", "sí es un sobre ajuste porque estoy contando dos veces lo mismo",
                         "no es sobreajuste (aprender ruido): es evidencia contada dos veces → exceso de confianza", en_ok="b"),
    T("y no «mejora»: la probabilidad sube sin información nueva (diapositiva 47)", 960, 760, tam=36, c="teal", w="sb", en="c"),
    sec="4. ¿Es sobreajuste?")

# ---- 6. Con los datos reales
v.b("¿Pasa esto con los correos reales? Un cálculo adicional. [[a]]Repetimos las cincuenta y cuatro señales de Spambase dos y cuatro veces, con Bernoulli y alfa uno, y los mismos pliegues. [[b]]El log loss fuera de pliegue pasa de 0.557 a 1.042 con dos repeticiones, y a 1.876 con cuatro. [[c]]Pero el AUC queda exactamente igual, 0.951: el orden de los correos no cambia, solo se exageran las probabilidades.",
    CL(), T("repetir las 54 señales de Spambase", 960, 150, tam=54, c="tinta", ent="subir"),
    tabla([["repeticiones", "log loss OOF", "AUC OOF"]] + [[str(d["k"]), f"{d['log_loss']:.3f}", f"{d['auc']:.3f}"] for d in DR],
          [600, 1000, 1360], 300, paso=110, tam=46, x0=360, x1=1600, ens=[None, "a", "b", "b"]),
    P("mismo orden (AUC igual), probabilidades exageradas", 960, 800, "violeta", 46, en="c"), adicional(1050, en="a"), sec="5. Con los datos reales")

# ---- 7. Calibración
GK = "gk10"
v.b("La otra cara de este problema es la calibración, en la diapositiva cuarenta y ocho. [[a]]Agrupamos los correos según su probabilidad fuera de pliegue, en intervalos de 0.1. Para cada intervalo comparamos la probabilidad media con la fracción de spam que realmente hubo. [[b]]Si el modelo estuviera bien calibrado, los puntos caerían sobre la diagonal. [[c]]Este es Bernoulli con alfa uno; el ganador de la selección usa 0.5, y no se ajustó ningún calibrador.",
    CL(), T("contrastar la probabilidad con lo que ocurrió · diapositiva 48", 960, 130, tam=46, c="tinta", ent="subir"),
    G(330, 230, 1110, 880, 0, 1, 0, 1, [dict(xs=[0, 1], ys=[0, 1], color="gris", grosor=3, revela=0.0),
                                       dict(xs=[c["prob_media"] for c in CB], ys=[c["fraccion"] for c in CB], color="azul", grosor=4, marcadores=12, linea=False, revela=0.0)],
      xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yfmt="{:.1f}", xlabel="probabilidad media",
      id=GK, tam_txt=26), T("fracción observada de spam", 330, 190, tam=30, c="tinta", w="sb", anc="lm"),
    U(GK, **{"series.1.revela": 1.0}, en="a", dur=1.5), U(GK, **{"series.0.revela": 1.0}, en="b", dur=1.0),
    T("Bernoulli, α = 1 (diagnóstico OOF)", 1500, 330, tam=36, c="azul", w="b", en="c"), diapo(48, en="a"), sec="6. Calibración")
v.b("Mira el intervalo de arriba, de 0.9 a uno. [[a]]Ahí caen mil ciento cincuenta y siete correos. La predicción media es 99.44 por ciento. [[b]]Pero el spam observado es mil sesenta y seis de mil ciento cincuenta y siete: 92.13 por ciento. [[c]]El modelo promete más certeza de la que cumple. Y el profesor lo notó en el gráfico: Bernoulli está bastante más cerca de la diagonal que la gaussiana.",
    R(C9["prob"], C9["frac"], 70, 70, "rojo", r=35, ref=GK, en="a"),
    T("intervalo 0.9–1: 1157 correos", 1500, 440, tam=36, c="tinta", w="sb", en="a"),
    T("predicción media: 99.44 %", 1500, 520, tam=38, c="rojo", w="b", en="a", t=0.5),
    T("spam observado: 1066/1157 ≈ 92.13 %", 1500, 600, tam=38, c="verde", w="b", en="b"),
    P("promete más certeza de la que cumple", 1500, 720, "ambar", 36, en="c"), grabacion("01:57:37", en="c", y=1010))
# ---- 8. Analogía
v.b("Una analogía. [[a]]Un amigo dice estar seguro al noventa y nueve por ciento cada vez que apuesta, [[b]]pero gana nueve de cada diez: acierta bastante, pero exagera. [[c]]Con él hay que descontar el entusiasmo. Eso es recalibrar.",
    CL(), T("el amigo que siempre está «seguro al 99 %»", 960, 150, tam=52, c="tinta", ent="subir"),
    P("dice: 99 %", 620, 380, "rojo", 56, en="a"), P("gana: 9 de cada 10", 1300, 380, "verde", 56, en="b"),
    P("acierta bastante, pero exagera: hay que descontar su entusiasmo", 960, 600, "violeta", 42, en="c"), analogia(1050), sec="7. Analogía: el amigo seguro")

# ---- 9. Un correo real: address
v.b("Ahora, explicar el correo de identificador uno, como en la diapositiva cuarenta y nueve. Empezamos por una señal: address, presente. [[a]]Entre los dos mil doscientos treinta legítimos de desarrollo, address aparece en doscientos veintiséis. Con alfa uno: doscientos veintisiete sobre dos mil doscientos treinta y dos, aproximadamente 0.1017. [[b]]Entre los mil cuatrocientos cincuenta spam, aparece en cuatrocientos noventa y ocho: cuatrocientos noventa y nueve sobre mil cuatrocientos cincuenta y dos, aproximadamente 0.3437. [[c]]Favorece al spam, pero aún faltan las otras cincuenta y tres señales.",
    CL(), T("reconstruir una señal del correo real · diapositiva 49", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["clase", "presencias", "total", "probabilidad suavizada"], ["legítimo", "226", "2230", "(226 + 1)/(2230 + 2) ≈ 0.1017"],
           ["spam", "498", "1450", "(498 + 1)/(1450 + 2) ≈ 0.3437"]], [360, 720, 1000, 1460], 300, paso=120, tam=42, x0=150, x1=1780,
          colores=[["tinta"] * 4, ["azul"] + ["tinta"] * 3, ["rojo"] + ["tinta"] * 3], ens=[None, "a", "b"]),
    P("address presente favorece spam: 0.3437 > 0.1017", 960, 700, "violeta", 46, en="c"),
    T("son probabilidades de una señal dentro de cada clase, no posteriores de spam", 960, 820, tam=34, c="gris", w="sb", en="c", t=1.0),
    diapo(49, en="a"), sec="8. Un correo real")

# ---- 10. Ocho señales
GA8 = "ga10"
TOP = I1["top"][::-1]
etq = {"$": "carácter $", "!": "carácter !"}
v.b("La diapositiva cincuenta muestra las ocho señales que más pesan en ese correo. [[a]]Siete están presentes: free, address, el signo de exclamación, email y otras tres palabras frecuentes en inglés. Una está ausente: el signo pesos. [[b]]Para cada señal, la barra azul es la probabilidad del estado observado dentro de los legítimos, y la naranja, dentro de los spam. [[c]]Por ejemplo, free presente: 0.0959 entre los legítimos, y 0.5558 entre los spam. Es lo que dijo el profesor: cincuenta y tanto por ciento dentro de spam, y casi 0.1 dentro de legítimo.",
    CL(), T("combinar las señales de un correo real · diapositiva 50", 960, 110, tam=46, c="tinta", ent="subir"),
    [x for i, r in enumerate(TOP) for x in (
        T(f"{etq.get(r['var'], r['var'])} ({'presente' if r['presente'] else 'ausente'})", 520, 210 + 88 * i, tam=30, c="tinta", w="sb", anc="rm", en="a", t=0.05 * i),
        Bx(540, 192 + 88 * i, 540 + 900 * r["p0"], 210 + 88 * i, "azul", None, 0, 6, grow="x", en="b", t=0.05 * i),
        Bx(540, 214 + 88 * i, 540 + 900 * r["p1"], 232 + 88 * i, "ambar", None, 0, 6, grow="x", en="b", t=0.05 * i))],
    T("dentro de legítimos", 1650, 300, tam=32, c="azul", w="b", en="b"), T("dentro de spam", 1650, 360, tam=32, c="ambar", w="b", en="b"),
    T("free: 0.0959 frente a 0.5558", 1650, 520, tam=34, c="tinta", w="b", en="c"),
    diapo(50, en="a"), grabacion("02:03:32", en="c", y=1010), sec="9. Las ocho señales")
v.b("Las cincuenta y cuatro señales, junto con las frecuencias de clase, producen una probabilidad de spam de 0.9964. [[a]]La diapositiva advierte: las alturas de las barras son factores; no se suman, ni representan efectos causales. [[b]]Y un cálculo adicional con logaritmos lo reconstruye: la previa aporta menos 0.430, porque hay menos spam que legítimos; las cincuenta y cuatro señales suman 6.065. [[c]]El total, 5.635, convertido a probabilidad, da 0.9964. Treinta y cuatro señales empujan hacia spam y veinte hacia legítimo.",
    CL(), T("de las señales a la probabilidad del correo", 960, 140, tam=52, c="tinta", ent="subir"),
    P("las barras son factores: no se suman ni son efectos causales", 960, 290, "ambar", 42, en="a"),
    T(f"previa: {I1['prior_log']:.3f}   +   señales: {I1['suma_aportes']:.3f}   =   {I1['log_odds']:.3f}", 960, 450, tam=48, c="tinta", w="b", en="b"),
    T("p = 1 / (1 + e^(−5.635)) ≈", 1000, 590, tam=50, c="tinta", w="sb", anc="rm", en="c"), N(I1["posterior"], 1030, 590, "{:.4f}", 70, "rojo", anc="lm", en="c", t=1.0),
    T("34 señales empujan hacia spam · 20 hacia legítimo", 960, 730, tam=40, c="teal", w="sb", en="c", t=2.0), adicional(1050, en="b"))

# ---- 11. Errores típicos
v.b("Errores típicos al leer estas salidas. [[a]]Uno: leer las barras de un correo como efectos causales de cada palabra. [[b]]Dos: creer que 0.9964 significa que el 99.64 por ciento de los correos parecidos son spam. Eso exige comprobar la calibración, y vimos que el modelo exagera. [[c]]Y tres: arreglar la dependencia con más suavizado. El suavizado evita ceros; la dependencia se ataca con la representación, o recalibrando.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("leer las barras como efectos causales", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("0.9964 = «99.64 % de los parecidos son spam»", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("arreglar la dependencia con más suavizado", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="10. Errores típicos")

# ---- 12. Piensa un segundo
v.b("Piensa un segundo. [[a]]En el experimento controlado, ¿qué probabilidad de spam da Naive Bayes con tres copias de la señal?",
    CL(), P("señal: en 80 de 100 spam y en 20 de 100 legítimos  ·  3 copias", 960, 330, "teal", 44, en="a"),
    P("P(spam) = ?", 960, 520, "violeta", 60, en="a", t=0.5), sec="11. Piensa un segundo", pausa=0.6)
v.b("La razón es cuatro al cubo: [[a]]sesenta y cuatro a uno. [[b]]La probabilidad es sesenta y cuatro sobre sesenta y cinco, aproximadamente 0.9846. Con la información real de una sola señal, debería seguir siendo 0.8.",
    CL(), T("4³ = 64 : 1  →  64/65 ≈", 1000, 360, tam=56, c="tinta", w="sb", anc="rm", en="a"), N(64 / 65, 1030, 360, "{:.4f}", 72, "rojo", anc="lm", en="b"),
    P("información real: una sola señal → 0.8", 960, 540, "azul", 46, en="b", t=1.0), ilustrativo(1050))

# ---- 13. Para llevar
v.b("Para llevarte: [[a]]si dos señales repiten información, Naive Bayes la cuenta dos veces: con una señal copiada, 0.80 pasa a 0.9412, y con cuatro copias, a 0.9961. [[b]]El orden de los correos no cambia, pero las probabilidades se vuelven extremas: en el intervalo de arriba, 99.44 por ciento predicho frente a 92.13 observado. [[c]]Y para explicar un correo, cada señal aporta un factor por clase: no un efecto causal.",
    CL(), P("señal copiada: 0.80 → 0.9412 → 0.9961", 960, 300, "rojo", 48, en="a"),
    P("calibración: 99.44 % predicho frente a 92.13 % observado", 960, 460, "ambar", 44, en="b"),
    P("cada señal aporta un factor por clase, no un efecto causal", 960, 620, "violeta", 44, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]pasar de la probabilidad a la acción. El umbral por costo, y la comprobación en los novecientos veintiuno correos históricos.",
    CL(), siguiente(11, "Decidir: umbral por costo y comprobación histórica"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
