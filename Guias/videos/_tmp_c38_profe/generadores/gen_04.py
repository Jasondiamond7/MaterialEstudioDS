import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, alerta, fila, codigo)

OUT = out("guion_c38_04_candidatos.json")
v = V("Ajustar no es seleccionar: los candidatos")
HP = D["h_part"]
VAL = [9.8, 10.4, 9.2, 10.1, 8.7, 10.3]          # diapositiva 14 (valores leidos del grafico; el 8.7 esta escrito)

v.portada("Ajustar no es seleccionar", "La receta completa y los siete candidatos",
          "Cuarta parada: la diferencia entre ajustar un modelo y seleccionar entre modelos. Y los siete candidatos que compara la clase.")

# ---- 1. Hook: el ganador tuvo suerte
v.b("Mira este ejemplo hipotético de la diapositiva catorce. [[a]]Seis modelos que, en verdad, son igual de buenos: todos tienen un error esperado de 10. [[b]]Pero en una muestra de validación, cada uno saca un error un poco distinto, por azar. [[c]]Elegimos el modelo cinco, porque obtuvo 8.7. ¿Ese 8.7 es el error que va a tener en casos nuevos? No: esa muestra lo favoreció. Su error esperado sigue siendo 10.",
    CL(), T("el menor error de validación puede ser optimista · diapositiva 14", 960, 140, tam=44, c="tinta", ent="subir"),
    G(400, 250, 1500, 820, 0.5, 6.5, 8, 11, [dict(xs=[0.5, 6.5], ys=[10, 10], color="tinta", grosor=4, revela=0.0, lapiz=False),
                                            dict(xs=[1, 2, 3, 4, 5, 6], ys=VAL, color="ambar", linea=False, marcadores=14, revela=0.0, lapiz=False)],
      xticks=[1, 2, 3, 4, 5, 6], yticks=[8, 9, 10, 11], yfmt="{:.0f}", xlabel="modelo candidato", ylabel="error", id="g14", tam_txt=28),
    U("g14", **{"series.0.revela": 1.0}, en="a", dur=0.8), T("error esperado = 10 para todos", 1490, 472, tam=30, c="tinta", w="sb", anc="rm", en="a", t=0.4),
    U("g14", **{"series.1.revela": 1.0}, en="b", dur=1.4),
    R(1225, 687, 90, 90, "rojo", r=45, en="c"), P("elegimos el 5: 8.7 · su error esperado es 10", 960, 920, "rojo", 44, en="c", t=0.6),
    diapo(14, en="a"), etiqueta("valores de los otros cinco leídos del gráfico", 1010, en="b"), sec="1. El ganador tuvo suerte")

# ---- 2. Analogía: la moneda
v.b("Es como seis personas que lanzan una moneda diez veces. [[a]]Una saca ocho caras y la declaramos «la mejor lanzadora». [[b]]Pero su moneda es igual a las otras: tuvo suerte. Si la hacemos lanzar de nuevo, lo más probable es que saque alrededor de cinco. [[c]]Cuando eliges al mejor entre varios, su resultado suele estar inflado. Por eso, para saber cuánto rendirá el elegido, hay que medirlo en datos que no participaron en la elección.",
    CL(), T("la mejor lanzadora de monedas", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "8 caras de 10", "la «mejor»… por azar", "a"), (1360, "gris", "otra vez", "≈ 5 caras: su moneda es igual", "b")],
             y=450, w=640, h=300, tam_t=52, tam_d=40),
    P("el elegido se mide con datos que no participaron en elegirlo", 960, 760, "violeta", 44, en="c"),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: la moneda")

# ---- 3. Ajuste y selección
v.b("La diapositiva quince separa dos actividades. [[a]]Ajustar, dentro de un candidato: estimar coeficientes, medias y escalas, usando solo las filas autorizadas. [[b]]Seleccionar, entre candidatos: elegir variables, transformaciones o la penalización, usando el error de validación. [[c]]Se elige el candidato con el menor error de validación cruzada.",
    CL(), T("qué se ajusta y qué se selecciona · diapositiva 15", 960, 150, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "ajuste", "coeficientes, medias y escalas, con las filas autorizadas", "a"),
              (1360, "violeta", "selección", "variables, transformaciones o penalización, con el error de validación", "b")],
             y=450, w=680, h=340, tam_t=52, tam_d=36),
    caja_formula("elegido = argmin  R_CV(m)", y=760, c="violeta", tam=50, w=800, en="c"), diapo(15, en="a"), sec="3. Ajustar y seleccionar")
v.b("Y aquí aparece una idea central de la clase: [[a]]cada candidato es una receta completa. [[b]]No es solo «una regresión»: es la regresión con su forma de estandarizar, sus términos y su penalización. [[c]]En la grabación se dijo así: cada candidato «es una receta completa». Validar un candidato exige ajustar la receta entera de nuevo en cada pliegue.",
    CL(), T("cada candidato es una receta completa", 960, 150, tam=54, c="tinta", ent="subir"),
    *[x for x in [Bc(400 + 380 * i, 420, 330, 150, "blanco", c, 5, 24, sombra=True, en="a" if i == 0 else "b", t=0.15 * i) for i, c in
                  enumerate(["azul", "violeta", "teal", "ambar"])]],
    *[T(t_, 400 + 380 * i, 420, tam=34, c=c, mw=300, en="a" if i == 0 else "b", t=0.15 * i + 0.1) for i, (t_, c) in
      enumerate([("estandarizar", "azul"), ("términos\ncuadráticos", "violeta"), ("estandarizar\nde nuevo", "teal"), ("ridge con α", "ambar")])],
    cita("es una receta completa", 640, en="c", tam=52), etiqueta("cita de la grabación de la clase", 720, en="c"),
    P("validar = reajustar la receta entera en cada pliegue", 960, 850, "violeta", 44, en="c", t=1.5))

# ---- 4. Alerta del material
v.b("Una corrección importante sobre la grabación. [[a]]Al explicar el ajuste, se dijo: «el ajuste dentro de los distintos candidatos, voy a realizarlo con el conjunto de prueba». [[b]]Es al revés: el ajuste se hace con las filas de entrenamiento autorizadas en cada pliegue. La diapositiva quince dice «usando las filas autorizadas». [[c]]Y el código ajusta con las filas que no son de validación. La prueba no se toca hasta el final; si se usara para ajustar, habría fuga de datos.",
    CL(), alerta(170), cita("el ajuste dentro de los distintos candidatos, voy a realizarlo con el conjunto de prueba", 320, en="a", tam=42),
    etiqueta("grabación de la clase", 410, en="a", t=0.4),
    P("se ajusta con las filas de ENTRENAMIENTO autorizadas · diapositiva 15", 960, 540, "verde", 42, en="b"),
    codigo(["ajuste = clone(modelo).fit(x.loc[~valida], y[~valida])"], y=690, w=1400, tam=36, en="c"),
    T("~valida = las filas del pliegue que NO se están validando", 960, 800, tam=34, c="gris", w="sb", en="c", t=0.6),
    P("la prueba se usa una sola vez, al final", 960, 910, "ambar", 44, en="c", t=1.6), sec="4. Una corrección")

# ---- 5. Los candidatos
v.b("Ahora, los candidatos de la diapositiva dieciséis. [[a]]Primero, el modelo constante: siempre predice la media del entrenamiento, que es 35.5 megapascales. [[b]]Segundo, la regresión lineal múltiple: un intercepto y un coeficiente por cada uno de los ocho predictores. [[c]]Tercero, la regresión cuadrática con penalización ridge.",
    CL(), T("qué modelos vamos a comparar · diapositiva 16", 960, 150, tam=48, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "constante", "ŷ = media del entrenamiento = 35.5 MPa", "a"),
              (960, "azul", "lineal", "ŷ = β₀ + Σ βⱼ xⱼ  (8 predictores)", "b"),
              (1520, "violeta", "cuadrática + ridge", "lineales, cuadrados e interacciones", "c")], y=470, w=500, h=340, tam_t=46, tam_d=36),
    real(1050, en="a", texto="media del entrenamiento recalculada con los datos de la clase"), sec="5. Los candidatos")
v.b("¿Cuántos términos tiene la cuadrática? [[a]]Ocho términos lineales, uno por predictor. [[b]]Ocho cuadrados, como cemento al cuadrado. [[c]]Y las interacciones entre pares, como cemento por agua: ocho por siete, sobre dos, son 28. En total: ocho más ocho más 28, igual a 44 términos.",
    CL(), T("44 términos", 960, 150, tam=56, c="tinta", ent="subir"),
    N(8, 360, 380, "{:.0f}", 110, "azul", en="a"), T("lineales", 360, 470, tam=34, c="gris", w="sb", en="a", t=0.3),
    N(8, 900, 380, "{:.0f}", 110, "violeta", en="b"), T("cuadrados (cemento²)", 900, 470, tam=34, c="gris", w="sb", en="b", t=0.3),
    N(28, 1460, 380, "{:.0f}", 110, "teal", en="c"), T("interacciones (cemento × agua)", 1460, 470, tam=34, c="gris", w="sb", en="c", t=0.3),
    T("8 · 7 / 2 = 28", 1460, 560, tam=38, c="teal", w="sb", en="c", t=1.2),
    P("8 + 8 + 28 = 44 términos", 960, 720, "violeta", 54, en="c", t=2.4), diapo(16, en="a"))

# ---- 6. Ridge
v.b("Con 44 términos, el ajuste puede volverse inestable. Para eso está ridge, en la diapositiva diecisiete. [[a]]Minimiza la suma de los errores al cuadrado, más alfa por la suma de los coeficientes al cuadrado. [[b]]Un alfa pequeño permite coeficientes grandes; un alfa grande los restringe con más fuerza. [[c]]El intercepto no se penaliza, y la escala de los términos importa: por eso se estandarizan antes.",
    CL(), T("cómo se ajusta la cuadrática: ridge · diapositiva 17", 960, 150, tam=48, c="tinta", ent="subir"),
    caja_formula("min  Σ (yᵢ − β₀ − zᵢᵀβ)²  +  α · Σ βⱼ²", y=320, c="ambar", tam=50, w=1300, en="a"),
    T("ajuste a los datos", 760, 410, tam=30, c="gris", w="sb", en="a", t=0.5), T("penalización", 1310, 410, tam=30, c="ambar", w="sb", en="a", t=0.8),
    P("α pequeño → coeficientes grandes", 560, 560, "azul", 42, en="b"), P("α grande → coeficientes restringidos", 1360, 560, "rojo", 42, en="b", t=0.6),
    T("el intercepto no se penaliza · la escala importa: se estandariza", 960, 700, tam=38, c="tinta", w="sb", en="c"),
    diapo(17, en="a"), sec="6. Ridge")
v.b("Un ejemplo pequeño, con un solo coeficiente, para ver el efecto. [[a]]Sin penalización, el coeficiente es la suma de x por y, sobre la suma de x al cuadrado. Si esas sumas son 20 y 10, da 2. [[b]]Con ridge, alfa se suma abajo: con alfa igual a 10, queda 20 sobre 20, igual a 1. [[c]]Y con alfa igual a mil, 20 sobre {1010|mil diez}: casi cero, 0.02. Alfa empuja el coeficiente hacia cero.",
    CL(), T("ridge con un solo coeficiente", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("β = Σ xy / (Σ x² + α)", y=300, c="ambar", tam=54, w=900),
    T("Σ xy = 20,  Σ x² = 10", 960, 420, tam=40, c="gris", w="sb"),
    T("α = 0:  20 / 10 =", 1000, 540, tam=44, c="tinta", w="sb", anc="rm", en="a"), N(2, 1030, 540, "{:.0f}", 60, "azul", anc="lm", en="a", t=0.6),
    T("α = 10:  20 / 20 =", 1000, 650, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(1, 1030, 650, "{:.0f}", 60, "violeta", anc="lm", en="b", t=0.6),
    T("α = 1000:  20 / 1010 ≈", 1000, 760, tam=44, c="tinta", w="sb", anc="rm", en="c"), N(0.02, 1030, 760, "{:.2f}", 60, "rojo", anc="lm", en="c", t=0.6),
    adicional(1050, en="a"))

# ---- 7. Siete candidatos y el código
v.b("En resumen, la clase compara siete candidatos. [[a]]Uno constante, uno lineal, y cinco cuadráticos con ridge: alfa igual a 0.1, a 1, a 10, a 100 y a mil. [[b]]Con cinco pliegues, son 35 ajustes de validación cruzada. [[c]]En el código, la receta cuadrática es una tubería: estandarizar, crear los términos de grado dos con Polynomial Features, estandarizar otra vez, y ajustar el modelo ridge.",
    CL(), T("siete candidatos · 35 ajustes", 960, 140, tam=54, c="tinta", ent="subir"),
    fila(270, [("constante", 300), ("lineal", 560), ("α = 0.1", 820), ("α = 1", 1040), ("α = 10", 1260), ("α = 100", 1480), ("α = 1000", 1700)],
         tams=[34] * 7, colores=["gris", "azul"] + ["violeta"] * 5, en="a", x0=180, x1=1820, fondo="linea", h=80),
    T("7 candidatos × 5 pliegues = 35 ajustes", 960, 390, tam=44, c="tinta", w="sb", en="b"),
    codigo(["make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),",
            "              StandardScaler(), Ridge(alpha=a, solver='svd'))"], y=600, w=1700, tam=34, en="c"),
    T("38_evaluacion_seleccion.py · sección 02", 960, 760, tam=30, c="gris", w="sb", en="c", t=0.6), diapo(23, en="b"), sec="7. Siete candidatos")

# ---- 8. El error típico
v.b("El error típico: [[a]]reportar el error de validación del ganador como si fuera su desempeño esperado. [[b]]Es el 8.7 de la diapositiva catorce: el número que usamos para elegir tiende a ser optimista, justamente porque lo elegimos por ser el más bajo. [[c]]Por eso existe la prueba reservada: un dato que no participó en la elección.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("reportar el error de validación del ganador", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("el 8.7 que sirvió para elegir… es optimista", 960, 440, tam=44, c="tinta", w="sb", en="b"),
    P("la prueba reservada no participa en la elección", 960, 620, "violeta", 48, en="c"), diapo(14, en="b"), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si tuvieras solo cuatro predictores, en vez de ocho, ¿cuántos términos tendría el modelo cuadrático completo, con lineales, cuadrados e interacciones?",
    CL(), T("4 predictores", 960, 300, tam=64, c="tinta", w="b", en="a"),
    P("lineales + cuadrados + interacciones = ?", 960, 480, "violeta", 52, en="a", t=0.8), sec="9. Piensa un segundo", pausa=0.6)
v.b("Cuatro lineales, [[a]]cuatro cuadrados, [[b]]y cuatro por tres, sobre dos, seis interacciones. [[c]]En total, 14 términos. Con ocho predictores eran 44: el número de términos crece rápido, y por eso la penalización ayuda.",
    CL(), N(4, 520, 360, "{:.0f}", 100, "azul"), T("lineales", 520, 450, tam=36, c="gris", w="sb"),
    N(4, 960, 360, "{:.0f}", 100, "violeta", en="a"), T("cuadrados", 960, 450, tam=36, c="gris", w="sb", en="a"),
    N(6, 1400, 360, "{:.0f}", 100, "teal", en="b"), T("4 · 3 / 2 interacciones", 1400, 450, tam=36, c="gris", w="sb", en="b"),
    P("4 + 4 + 6 = 14 términos", 960, 640, "violeta", 54, en="c"), ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]ajustar es estimar dentro de un candidato, con las filas autorizadas; seleccionar es comparar candidatos con el error de validación. [[b]]Cada candidato es una receta completa, que se reajusta entera en cada pliegue. [[c]]Y el error del ganador en validación es optimista: su desempeño se mide en la prueba.",
    CL(), P("ajustar ≠ seleccionar", 960, 300, "azul", 52, en="a"),
    P("candidato = receta completa, reajustada en cada pliegue", 960, 460, "violeta", 46, en="b"),
    P("el error del ganador en validación es optimista", 960, 620, "ambar", 46, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]el sobreajuste, y las tres fuentes del error: sesgo, varianza y ruido.",
    CL(), siguiente(5, "Sobreajuste, sesgo y varianza"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
