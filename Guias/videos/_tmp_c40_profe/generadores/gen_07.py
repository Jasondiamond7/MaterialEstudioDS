import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_07_spambase.json")
v = V("Spambase: 54 señales")
S = D["spam"]
PA = S["particion"]

v.portada("Spambase: 54 señales", "Del ejemplo construido a 4601 correos reales",
          "Séptima parada. Dejamos los cien correos construidos y pasamos a {4601|cuatro mil seiscientos un} correos reales de Spambase. Vamos a ver qué contiene cada fila, cómo se separan los datos y qué probabilidad recibe un correo concreto.")

# ---- 1. Qué contiene
v.b("La diapositiva treinta y uno presenta los datos. [[a]]Cuarenta y ocho frecuencias de palabras, que describen el contenido. [[b]]Seis frecuencias de caracteres, como el signo de exclamación o el signo pesos. [[c]]Y la etiqueta de cada correo: spam o legítimo. En total, cincuenta y cuatro señales. El archivo trae además tres medidas de mayúsculas, que se excluyen del modelo.",
    CL(), T("construyamos el filtro con 4601 correos · diapositiva 31", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["qué contiene", "qué utilizaremos", "para qué"], ["48 frecuencias de palabras", "las 48 columnas", "describir el contenido"],
           ["6 frecuencias de caracteres", "las 6 columnas", "describir otras señales"], ["etiqueta de cada correo", "spam o legítimo", "aprender por clase"]],
          [480, 1040, 1500], 290, paso=110, tam=40, x0=170, x1=1760, ens=[None, "a", "b", "c"]),
    P("54 frecuencias; se excluyen las 3 medidas de mayúsculas", 960, 800, "ambar", 44, en="c", t=1.5),
    diapo(31, en="a"), sec="1. Qué contiene")
v.b("Sobre la fuente: [[a]]es el conjunto Spambase, de Hopkins y coautores, de mil novecientos noventa y nueve, publicado en el repositorio de la UCI con licencia abierta. [[b]]Son correos históricos, de fines de los noventa: el mismo conjunto que usamos en la clase treinta y ocho para evaluar clasificadores. [[c]]Y la pregunta es la misma de siempre: dadas las señales de un correo, ¿qué probabilidad de spam le asigna el modelo?",
    CL(), T("la fuente", 960, 150, tam=56, c="tinta", ent="subir"),
    P("Hopkins et al. (1999), Spambase, UCI · CC BY 4.0", 960, 320, "teal", 46, en="a"),
    P("corpus histórico · el mismo de la Clase 38", 960, 480, "gris", 46, en="b"),
    T("dadas las señales de un correo, ¿qué probabilidad de spam le asigna el modelo?", 960, 680, tam=44, c="tinta", w="sb", mw=1500, en="c"),
    diapo(31, en="a"))

# ---- 2. Una fila
v.b("La diapositiva treinta y dos muestra una fila: el correo con identificador uno, el menor del conjunto de desarrollo. [[a]]La palabra free: 0.32 por ciento de las palabras; presencia, uno. [[b]]La palabra money: cero por ciento; presencia, cero. [[c]]El carácter de exclamación: 0.778 por ciento de los caracteres; presencia, uno. Ojo con la escala: el archivo guarda 0.32 para representar 0.32 por ciento.",
    CL(), T("una fila describe las señales de un correo · diapositiva 32", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["señal", "frecuencia registrada", "presencia: frecuencia > 0"], ["palabra free", "0.32 % de las palabras", "1: presente"],
           ["palabra money", "0 % de las palabras", "0: ausente"], ["carácter !", "0.778 % de los caracteres", "1: presente"]],
          [420, 980, 1500], 290, paso=110, tam=40, x0=170, x1=1760,
          colores=[["tinta"] * 3, ["tinta", "tinta", "verde"], ["tinta", "tinta", "gris"], ["tinta", "tinta", "verde"]], ens=[None, "a", "b", "c"]),
    P("el CSV guarda 0.32 para representar 0.32 %", 960, 800, "ambar", 44, en="c", t=1.5),
    diapo(32, en="a"), sec="2. Una fila")
v.b("Dos detalles más. [[a]]La predicción usa solo las señales: el identificador y la etiqueta del correo no son entradas del modelo. [[b]]Y un dato adicional: de las cincuenta y cuatro señales de este correo, nueve están presentes y cuarenta y cinco ausentes. Para Bernoulli, las cuarenta y cinco ausencias también cuentan.",
    CL(), P("entradas: solo las señales (ni el identificador ni la etiqueta)", 960, 320, "violeta", 44, en="a"),
    T("correo de identificador 1: 9 señales presentes y 45 ausentes", 960, 500, tam=48, c="tinta", w="b", en="b"),
    P("para Bernoulli, las ausencias también cuentan", 960, 650, "teal", 46, en="b", t=1.0), adicional(1050, en="b"))

# ---- 3. Leer los datos
v.b("En código, la diapositiva treinta y tres lee el archivo y separa entradas de respuesta. [[a]]Las variables se eligen por el nombre: las que empiezan con el prefijo de frecuencia de palabra o con el de frecuencia de carácter. [[b]]Se toma el conjunto de entrenamiento, se arma la matriz X con esas columnas y el vector y con la etiqueta. [[c]]Resultado: X tiene tres mil seiscientos ochenta correos y cincuenta y cuatro frecuencias. Elegir por nombre deja fuera identificadores, grupos, pliegues y la propia respuesta.",
    CL(), T("leer los datos y separar entradas de respuesta · diapositiva 33", 960, 140, tam=46, c="tinta", ent="subir"),
    codigo(['datos = pd.read_csv(DATOS/"spam_s40.csv")', 'variables = [c for c in datos', '    if c.startswith(("word_freq_", "char_freq_"))]',
            'ent = datos[datos.conjunto == "entrenamiento"]', 'X = ent[variables].to_numpy()', 'y = ent.spam.to_numpy()'],
           y=400, w=1500, tam=34, c="violeta", en="a"),
    P("X: 3680 correos × 54 frecuencias  ·  y: 1 spam, 0 legítimo", 960, 760, "teal", 44, en="c"),
    diapo(33, en="a"), sec="3. Leer los datos")

# ---- 4. La partición
v.b("La diapositiva treinta y cuatro conserva la partición guardada en los datos. [[a]]Desarrollo: tres mil seiscientos ochenta correos, para aprender y comparar recetas. [[b]]Prueba histórica: novecientos veintiuno, para comprobar la receta elegida. [[c]]Contando los datos: en desarrollo hay dos mil doscientos treinta legítimos y mil cuatrocientos cincuenta spam, un 39.4 por ciento de spam; en la prueba, quinientos cincuenta y ocho y trescientos sesenta y tres.",
    CL(), T("aprender con unos correos y predecir otros · diapositiva 34", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["conjunto", "correos", "legítimos", "spam", "uso"], ["desarrollo", "3680", str(PA["entrenamiento_0"]), str(PA["entrenamiento_1"]), "aprender y comparar"],
           ["prueba histórica", "921", str(PA["prueba_0"]), str(PA["prueba_1"]), "comprobar la elegida"]],
          [330, 680, 930, 1160, 1530], 310, paso=120, tam=40, x0=130, x1=1790, ens=[None, "a", "b"]),
    P("desarrollo: 1450 / 3680 ≈ 39.4 % de spam", 960, 680, "ambar", 44, en="c"), diapo(34, en="a"), real(1010, en="c"), sec="4. La partición")
v.b("Dentro del desarrollo, se usan cinco pliegues. [[a]]En cada vuelta se aprende con cuatro y se predice el restante. Al repetirlo cinco veces, cada correo recibe una predicción fuera de pliegue, que abreviamos OOF. [[b]]Un detalle de diseño: las filas con las mismas cincuenta y siete entradas originales se mantienen juntas en el mismo pliegue. En los datos hay {4601|cuatro mil seiscientos un} filas, pero solo cuatro mil doscientos siete grupos distintos. [[c]]Así, un correo repetido no puede estar a la vez en el ajuste y en la validación.",
    CL(), T("cinco pliegues y grupos de filas idénticas", 960, 150, tam=54, c="tinta", ent="subir"),
    casillas(["1", "2", "3", "4", "5"], 320, paso=200, w=170, h=110, c="azul", en="a", pre="pl", colores=["azul", "azul", "azul", "azul", "ambar"]),
    T("aprender con 4 pliegues  ·  predecir el restante  ·  repetir 5 veces", 960, 450, tam=40, c="tinta", w="sb", en="a", t=1.0),
    T("4601 filas  ·  4207 grupos de filas con las mismas 57 entradas", 960, 600, tam=42, c="teal", w="b", en="b"),
    P("un correo repetido no queda a la vez en ajuste y validación", 960, 750, "violeta", 44, en="c"), diapo(34, en="a"),
    adicional(1010, en="b", texto="Conteo adicional de grupos con el archivo spam_s40.csv"))
v.b("Y una advertencia honesta de la diapositiva: [[a]]los novecientos veintiuno correos de prueba ya se usaron al evaluar otros clasificadores, en la clase treinta y ocho. [[b]]Por eso la comprobación final es histórica, no una nueva prueba independiente.",
    CL(), P("los 921 ya se usaron al evaluar otros clasificadores", 960, 360, "ambar", 48, en="a"),
    P("comprobación histórica, no prueba independiente", 960, 540, "rojo", 48, en="b"), diapo(34, en="a"))

# ---- 5. Del correo real a su probabilidad
v.b("Ahora, la primera predicción real, de la diapositiva treinta y cinco. [[a]]La función ajustar predecir, incluida en los dos scripts, recibe los datos de ajuste y los correos por predecir. Usamos Bernoulli con alfa igual a uno. [[b]]Internamente transforma cada frecuencia positiva en presencia y combina las cincuenta y cuatro señales. [[c]]Para el correo de identificador uno, la probabilidad de spam es aproximadamente 0.9964.",
    CL(), T("del correo real a su probabilidad de spam · diapositiva 35", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["i = int(np.argmin(ent.id.to_numpy()))", "correo = X[i:i+1]", 'p, modelo, logp = ajustar_predecir(X, y, correo, "bern", 1)', "print(p[0])"],
           y=380, w=1500, tam=36, c="violeta", en="a"),
    T("p ≈", 1000, 660, tam=60, c="tinta", w="sb", anc="rm", en="c"), N(D["id1"]["posterior"], 1030, 660, "{:.4f}", 80, "rojo", anc="lm", en="c", t=0.5),
    diapo(35, en="a"), sec="5. La primera predicción real")
v.b("Una precisión sobre la grabación. [[a]]Al mostrar este resultado, cerca del minuto noventa y dos, se leyó una probabilidad de 0.9954. [[b]]La diapositiva y el script dan 0.9964; recalculado con el script, 0.996442.",
    CL(), error_material("01:32:34", "para el ID1 por ejemplo vuelve con una probabilidad de 0.9954", "diapositiva 35 y script: p ≈ 0.9964 (0.996442)", en_ok="b"),
    sec="6. Una precisión")
v.b("Y una advertencia importante de la misma diapositiva. [[a]]Este correo participa en el ajuste: el modelo aprendió con él. [[b]]Por eso el 0.9964 ilustra una predicción, pero no evalúa nada. La calidad del filtro se va a medir con las probabilidades fuera de pliegue.",
    CL(), P("el correo participó en el ajuste", 960, 360, "ambar", 52, en="a"),
    P("ilustra una predicción · no evalúa el filtro", 960, 540, "violeta", 50, en="b"), diapo(35, en="a"))

# ---- 6. Presencia o frecuencia
v.b("La diapositiva treinta y seis hace una pregunta clave: ¿basta con saber que una palabra está presente? [[a]]Con presencia, free entra como uno, porque 0.32 es mayor que cero: solo sabemos que la palabra aparece. [[b]]Con frecuencia, entra como 0.32: sabemos qué proporción de las palabras corresponde a free. [[c]]Bernoulli trata igual cualquier frecuencia positiva: 0.01 o diez da lo mismo.",
    CL(), T("¿basta con saber que una palabra está presente? · diapositiva 36", 960, 140, tam=46, c="tinta", ent="subir"),
    tabla([["representación", "valor que entra", "información conservada"], ["presencia", "1, porque 0.32 > 0", "la palabra aparece"],
           ["frecuencia", "0.32", "qué proporción de palabras es free"]], [400, 900, 1430], 320, paso=120, tam=42, x0=130, x1=1810, ens=[None, "a", "b"]),
    P("Bernoulli trata igual cualquier frecuencia positiva", 960, 720, "ambar", 46, en="c"), diapo(36, en="a"), sec="7. Presencia o frecuencia")
v.b("Una analogía. [[a]]Presencia es preguntar: ¿tienes fiebre, sí o no? [[b]]Frecuencia es mirar el termómetro: treinta y siete y medio o cuarenta. [[c]]El termómetro da más información, pero para usarla hay que suponer cómo se distribuyen las temperaturas en cada clase. Y la diapositiva lo dice así: conservar más detalle exige otro supuesto, y hay que comprobar si ese detalle mejora las probabilidades.",
    CL(), T("¿fiebre sí o no, o el termómetro?", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(620, "verde", "presencia", "¿fiebre? sí / no", "a"), (1300, "rojo", "frecuencia", "37.5 °C o 40 °C", "b")], y=420, w=560, h=260, tam_t=52, tam_d=44),
    P("más detalle exige otro supuesto: hay que comprobar si mejora", 960, 700, "violeta", 44, en="c"), analogia(1050), sec="8. Analogía: el termómetro")

# ---- 7. Error típico
v.b("Errores típicos con datos reales. [[a]]Uno: dejar entrar el identificador, el pliegue o la propia etiqueta como si fueran señales. Seleccionar las columnas por nombre lo evita. [[b]]Dos: evaluar el filtro con un correo que participó en el ajuste, como el de identificador uno. [[c]]Y tres: mirar los correos de prueba mientras todavía se están comparando recetas.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("usar identificador, pliegue o etiqueta como señal", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("evaluar con un correo que participó en el ajuste", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("mirar la prueba mientras se comparan recetas", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="9. Errores típicos")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]En un correo de doscientas palabras, la frecuencia de free es 1.5 por ciento. [[b]]¿Cuántas veces aparece la palabra, y qué valor entra en el modelo de Bernoulli?",
    CL(), P("200 palabras  ·  free = 1.5 %", 960, 330, "teal", 56, en="a"),
    P("¿cuántas veces aparece? · ¿qué entra en Bernoulli?", 960, 520, "violeta", 50, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("El 1.5 por ciento de doscientas palabras [[a]]son tres apariciones. [[b]]Pero en Bernoulli entra solo un uno: presente. Fíjate que para saber las tres apariciones necesitamos la longitud del correo; con el porcentaje solo, no se puede. Eso va a importar en la próxima parada.",
    CL(), T("1.5 % × 200 = 3 apariciones", 960, 340, tam=60, c="tinta", w="b", en="a"),
    T("Bernoulli: 1 (presente)", 960, 480, tam=56, c="verde", w="b", en="b"),
    P("sin la longitud del correo no se recuperan las apariciones", 960, 640, "ambar", 44, en="b", t=1.5), ilustrativo(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]Spambase tiene {4601|cuatro mil seiscientos un} correos y cincuenta y cuatro frecuencias; se aprende con tres mil seiscientos ochenta y se comprueba con novecientos veintiuno. [[b]]Bernoulli convierte cada frecuencia positiva en presencia. [[c]]Y el correo de identificador uno recibe 0.9964 de probabilidad de spam, pero participó en el ajuste: ilustra, no evalúa.",
    CL(), P("4601 correos · 54 frecuencias · 3680 desarrollo + 921 prueba", 960, 300, "violeta", 44, en="a"),
    P("Bernoulli: frecuencia > 0 → presencia", 960, 460, "teal", 48, en="b"),
    P("identificador 1: p ≈ 0.9964, pero participó en el ajuste", 960, 620, "ambar", 44, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]las variantes. La gaussiana, que usa la frecuencia transformada, y la multinomial, que cuenta palabras.",
    CL(), siguiente(8, "Variantes: Bernoulli, gaussiana y multinomial"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
