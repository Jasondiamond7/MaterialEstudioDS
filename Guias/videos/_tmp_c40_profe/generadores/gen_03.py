import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_03_senales.json")
v = V("Dos señales en un mismo correo")

v.portada("Dos señales en un mismo correo", "Bernoulli y la explosión de combinaciones",
          "Tercera parada. Un correo real tiene muchas señales, no una. Vamos a agregar la segunda, el enlace, y a ver por qué contar todas las combinaciones se vuelve imposible muy rápido.")

# ---- 1. La segunda señal
v.b("La diapositiva once agrega un segundo indicador a los mismos cien correos: si el correo trae un enlace. [[a]]Entre los ochenta legítimos, ocho traen oferta y dieciséis traen enlace. [[b]]Entre los veinte spam, dieciséis traen oferta y catorce traen enlace. [[c]]Y el correo que queremos clasificar tiene oferta presente y enlace ausente. Lo escribimos como equis igual a uno, cero.",
    CL(), T("dos señales en un mismo correo · diapositiva 11", 960, 140, tam=50, c="tinta", ent="subir"),
    tabla([["clase", "correos", "con oferta", "con enlace"], ["legítimo", "80", "8", "16"], ["spam", "20", "16", "14"]],
          [480, 860, 1200, 1520], 300, paso=110, tam=46, x0=240, x1=1700,
          colores=[["tinta"] * 4, ["azul"] + ["tinta"] * 3, ["rojo"] + ["tinta"] * 3], ens=[None, "a", "b"]),
    P("correo nuevo: x = (1, 0) → oferta presente, enlace ausente", 960, 700, "violeta", 46, en="c"),
    diapo(11, en="a"), sec="1. La segunda señal")
v.b("Fíjate en lo que esta tabla no dice. [[a]]Sabemos cuántos legítimos traen oferta y cuántos traen enlace, pero no sabemos si son los mismos correos. [[b]]Como dijo el profesor, puede que esos ocho con oferta no tengan el enlace; no lo sabemos, solo sabemos la frecuencia de cada señal por separado.",
    CL(), T("la tabla da cada señal por separado", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(620, "azul", "8 con oferta", "¿cuántos de ellos traen enlace?", "a"), (1300, "azul", "16 con enlace", "¿cuántos de ellos traen oferta?", "a")],
             y=440, w=560, h=280, tam_t=54, tam_d=38),
    cita("yo no sé de esta tabla descriptiva si es que estos ocho de oferta pertenecen a los ocho con el enlace", y=780, tam=36, en="b"),
    grabacion("00:32:47", en="b"))

# ---- 2. Alerta del material
v.b("Una aclaración sobre la grabación. [[a]]Al presentar esta tabla, cerca del minuto treinta y dos, se dijo que dentro de spam, catorce no tienen enlace. [[b]]La diapositiva dice lo contrario: catorce spam tienen enlace, y por lo tanto seis no lo tienen. Unos minutos después, en la clase se usa bien: catorce con enlace, seis sin enlace.",
    CL(), error_material("00:32:22", "dentro de spam 14 no tienen enlace", "diapositiva 11: 14 spam con enlace → 20 − 14 = 6 sin enlace", en_ok="b"),
    T("más adelante (00:34:55) se usa bien: enlace presente en 14 de los 20 spam", 960, 780, tam=36, c="teal", w="sb", en="b", t=2.0),
    sec="2. Una aclaración")

# ---- 3. Bernoulli
v.b("Cada señal responde una pregunta con dos respuestas: ¿aparece oferta?, ¿hay enlace? [[a]]Una variable de Bernoulli representa justamente un resultado sí o no. Usamos uno para presente y cero para ausente. [[b]]No nos importa cuántas veces aparece la palabra, ni cuántos enlaces hay: solo si está o no está.",
    CL(), T("presente o ausente: aquí aparece Bernoulli · diapositiva 12", 960, 140, tam=48, c="tinta", ent="subir"),
    tarjetas([(620, "verde", "1 = presente", "la señal aparece en el correo", "a"), (1300, "gris", "0 = ausente", "la señal no aparece", "a")],
             y=440, w=560, h=280, tam_t=56, tam_d=38),
    P("no importa cuántas veces: solo si aparece o no", 960, 720, "violeta", 46, en="b"), diapo(12, en="a"), sec="3. Bernoulli")
v.b("Con los veinte spam del ejemplo: [[a]]el enlace está presente en catorce: catorce sobre veinte, 0.70. [[b]]Y ausente en veinte menos catorce, seis: seis sobre veinte, 0.30. [[c]]Dentro de la misma clase, presente y ausente sí suman uno. Una sola probabilidad describe la señal: la de presencia. La de ausencia es uno menos esa.",
    CL(), T("en los 20 spam del ejemplo", 960, 150, tam=54, c="tinta", ent="subir"),
    tabla([["enlace", "conteo", "probabilidad"], ["presente (1)", "14", "14/20 = 0.70"], ["ausente (0)", "20 − 14 = 6", "6/20 = 0.30"]],
          [520, 960, 1400], 300, paso=120, tam=46, x0=260, x1=1660, ens=[None, "a", "b"]),
    P("0.70 + 0.30 = 1   ·   ausencia = 1 − presencia", 960, 720, "teal", 46, en="c"), diapo(12, en="a"))
v.b("Un detalle que la diapositiva subraya. [[a]]Un cero significa ausencia observada de la señal: miramos el correo y el enlace no estaba. [[b]]No significa dato faltante. Si no supiéramos si el correo trae enlace, eso sería otra situación, y el cálculo sería distinto.",
    CL(), P("0 = ausencia observada", 960, 340, "verde", 60, en="a"),
    P("0 ≠ dato faltante", 960, 520, "rojo", 60, en="b"), diapo(12, en="a"))

# ---- 4. Analogía
v.b("Una analogía. [[a]]Piensa en un formulario de casillas: ¿trae oferta?, ¿trae enlace?, ¿trae la palabra gratis? Cada casilla se marca sí o no. [[b]]Para cada clase aprendemos qué tan seguido se marca cada casilla. Ese es el modelo de Bernoulli: una probabilidad de sí por casilla y por clase.",
    CL(), T("un formulario de casillas", 960, 150, tam=56, c="tinta", ent="subir"),
    K(560, 330, "check", 56, en="a"), T("¿trae oferta?", 640, 330, tam=46, c="tinta", w="sb", anc="lm", en="a"),
    K(560, 440, "cruz", 56, en="a"), T("¿trae enlace?", 640, 440, tam=46, c="tinta", w="sb", anc="lm", en="a"),
    K(560, 550, "check", 56, en="a"), T("¿trae la palabra gratis?", 640, 550, tam=46, c="tinta", w="sb", anc="lm", en="a"),
    P("por clase: qué tan seguido se marca «sí» en cada casilla", 960, 760, "violeta", 44, en="b"), analogia(1050), sec="4. Analogía: un formulario")

# ---- 5. Contar combinaciones
v.b("Ahora, la pregunta de la diapositiva trece: ¿podemos contar todas las combinaciones? [[a]]Con dos señales hay cuatro tipos de correo: sin oferta y sin enlace, sin oferta y con enlace, con oferta y sin enlace, y con oferta y con enlace. [[b]]Para usar la cuenta directa de la parada uno, necesitaríamos saber, en cada clase, cuántos correos hay de cada tipo. Y esa tabla no la tenemos.",
    CL(), T("¿podemos contar todas las combinaciones? · diapositiva 13", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["oferta", "enlace", "legítimos", "spam"], ["no", "no", "?", "?"], ["no", "sí", "?", "?"], ["sí", "no", "?", "?"], ["sí", "sí", "?", "?"]],
          [560, 820, 1160, 1440], 280, paso=96, tam=44, x0=380, x1=1620, ens=[None, "a", "a", "a", "a"]),
    R(960, 568, 1280, 86, "violeta", r=20, en="b"), diapo(13, en="a"), sec="5. Contar combinaciones")
v.b("Y el problema crece muy rápido. [[a]]Cada señal nueva duplica las combinaciones. Con tres señales, ocho; con diez, dos elevado a diez: mil veinticuatro patrones posibles por clase. [[b]]Con las cincuenta y cuatro señales de Spambase, dos elevado a cincuenta y cuatro: unos dieciocho mil billones de patrones. [[c]]Y en los datos reales tenemos tres mil seiscientos ochenta correos para aprender.",
    CL(), T("cada señal duplica las combinaciones", 960, 150, tam=54, c="tinta", ent="subir"),
    tabla([["señales", "patrones por clase"], ["2", "4"], ["3", "8"], ["10", "2¹⁰ = 1024"], ["54", "2⁵⁴ ≈ 1.8 × 10¹⁶"]],
          [700, 1220], 280, paso=100, tam=46, x0=420, x1=1500, ens=[None, "a", "a", "a", "b"]),
    P("datos de desarrollo: 3680 correos", 960, 860, "ambar", 46, en="c"), diapo(13, en="a"),
    adicional(1010, en="b", texto="Cálculo adicional: 2⁵⁴ para las 54 señales de Spambase"))

# ---- 6. Patrones sin observar
v.b("¿Qué pasa con tantos patrones? [[a]]La mayoría nunca aparece en los datos. Y si un patrón no aparece, su frecuencia es cero, y la probabilidad estimada también es cero. [[b]]El profesor lo explicó como el argumento frecuentista: estamos usando la frecuencia observada como probabilidad. [[c]]Con pocos correos, muchos patrones quedan sin observar, y un correo nuevo con un patrón nunca visto quedaría sin respuesta.",
    CL(), T("con pocos correos, muchos patrones quedan sin observar", 960, 150, tam=50, c="tinta", ent="subir"),
    flujo([("patrón no observado", "gris", "a"), ("frecuencia 0", "ambar", "a"), ("probabilidad 0", "rojo", "a")], y=420, w=420, h=200, x0=440, paso=520, tam=44),
    cita("si no pasó en mi datos entonces la frecuencia es cero, y por consecuencia la probabilidad es cero", y=700, tam=38, en="b"),
    grabacion("00:39:11", en="b"), sec="6. Patrones sin observar")
v.b("Y hay otra forma de verlo, contando parámetros. Es un cálculo adicional. [[a]]La tabla completa de diez señales necesita, en cada clase, mil veintitrés probabilidades: una por patrón, menos una porque todas suman uno. [[b]]Si en cambio aprendemos cada señal por separado, bastan diez probabilidades por clase, una por señal. [[c]]Esa es exactamente la propuesta de la diapositiva trece: combinar lo que aprendimos de cada señal por separado.",
    CL(), T("cuántas probabilidades hay que aprender (10 señales)", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "1023", "tabla completa: una por patrón (menos una)", "a"), (1300, "verde", "10", "una por señal, por separado", "b")],
             y=460, w=560, h=320, tam_t=96, tam_d=38),
    P("Naive Bayes: combinar lo aprendido de cada señal por separado", 960, 780, "violeta", 44, en="c"),
    adicional(1050, en="a"), diapo(13, en="c", y=1010))

# ---- 7. Error típico
v.b("Errores típicos en esta parada. [[a]]Uno: leer el cero como dato faltante. El cero es una observación: la señal no estaba. [[b]]Dos: suponer que las dos columnas de la tabla nos dicen cuántos correos tienen ambas señales. No lo dicen: cada columna es una frecuencia por separado. [[c]]Y tres: creer que con más señales siempre podremos contar los patrones. Con cincuenta y cuatro señales, ni con millones de correos.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("leer el 0 como dato faltante", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("deducir de las columnas cuántos tienen ambas señales", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("creer que siempre se pueden contar los patrones", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="7. Errores típicos")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si agregamos una tercera señal, la palabra cuenta, [[b]]¿cuántos patrones hay por clase, y cuántas probabilidades de presencia necesitamos si aprendemos cada señal por separado?",
    CL(), P("señales: oferta, enlace y cuenta", 960, 330, "teal", 52, en="a"),
    P("¿patrones por clase?  ·  ¿probabilidades por separado?", 960, 520, "violeta", 50, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Tres señales dan dos elevado a tres: [[a]]ocho patrones por clase. [[b]]Pero por separado bastan tres probabilidades de presencia por clase, una por señal. La diferencia, que aquí es pequeña, se vuelve gigantesca con decenas de señales.",
    CL(), T("2³ = 8 patrones por clase", 960, 340, tam=60, c="rojo", w="b", en="a"),
    T("3 probabilidades por clase, una por señal", 960, 500, tam=56, c="verde", w="b", en="b"), ilustrativo(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]cada señal es una variable de Bernoulli: uno si está presente, cero si está ausente, y el cero es una observación. [[b]]Dentro de una clase, una sola probabilidad la describe: en los spam, enlace presente 0.70 y ausente 0.30. [[c]]Y contar todos los patrones es imposible: cada señal duplica las combinaciones. Por eso necesitamos un atajo.",
    CL(), P("Bernoulli: 1 presente · 0 ausente (observado)", 960, 300, "violeta", 46, en="a"),
    P("en los spam: enlace 14/20 = 0.70 · sin enlace 6/20 = 0.30", 960, 460, "teal", 44, en="b"),
    P("cada señal duplica los patrones: 2¹⁰ = 1024", 960, 620, "ambar", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el atajo. El supuesto ingenuo de independencia dentro de cada clase, y la cuenta completa del correo con oferta y sin enlace.",
    CL(), siguiente(4, "El supuesto naive: independencia dentro de cada clase"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
