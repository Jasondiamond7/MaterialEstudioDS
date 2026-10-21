"""Apuntes de lectura de la sesión 0.1: explicación ordenada + ejemplos resueltos + pistas."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, SOFT,
                           band, box, build, bullets, img, numbered, table)

E0 = colors.HexColor("#475569")
BLUE = colors.HexColor("#2563EB")
BLUE_BG = colors.HexColor("#EFF6FF")


def h(num, title):
    return P(f'<font color="#475569">{num}</font>  {title}', "h2")


def para(text):
    return [P(text), Spacer(1, 4)]


def ejemplo(title, flow):
    return box([P(f"<b>Ejemplo resuelto · {title}</b>", "cellb"), Spacer(1, 3)] + flow,
               bg=SOFT, border=E0)


def pista(n, flow):
    return box([P(f"<b>Pista para el Ejercicio {n} del notebook</b>", "cellb"), Spacer(1, 2)] + flow,
               bg=BLUE_BG, border=BLUE)


def recuerda(items):
    return box([P("<b>Lo que debes recordar</b>", "cellb"), Spacer(1, 2),
                bullets(items, "cell", GREEN)], bg=GREEN_BG, border=GREEN)


def code(lines):
    return box([P(l, "mono") for l in lines], bg=colors.white, width=CONTENT_W - 22)


def apuntes():
    s = []
    s.append(band("Apuntes · Sesión 0.1 · Describir los datos",
                  "Para leer hoy y resolver el notebook 0.1 · lectura de ~40 min", E0))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 5 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("El tipo manda", "Antes de calcular, decide si cada variable es numérica o categórica. Eso define qué puedes hacer con ella."),
                      ("Cuenta los vacíos", "Antes de analizar, cuenta los datos faltantes. pandas los ignora en silencio."),
                      ("Media vs mediana", "La media usa todos los valores y los extremos la arrastran. La mediana es el valor del medio y no le afectan."),
                      ("Dispersión", "La desviación estándar es la distancia típica a la media, en las unidades de los datos. Con muestras, se divide por n − 1."),
                      ("Boxplot", "Resume 5 números y marca como outlier lo que está a más de 1.5·IQR fuera de la caja."),
                  ], E0)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Cómo leer estos apuntes: cada sección tiene la explicación, un <b>ejemplo resuelto</b> con números distintos "
               "a los del notebook (para que practiques el método, no copies la respuesta) y una <b>pista</b> para el "
               "ejercicio correspondiente.", "small"))

    # ------------------------------------------------------------ 1
    s.append(h("1", "Tipos de variables"))
    s += para("Cada columna de una tabla es una <b>variable</b>. Lo primero es decidir de qué tipo es, porque eso define "
              "qué cálculos tienen sentido. Si sacas el promedio de «industria», el resultado es absurdo. Y un gráfico de "
              "barras del colesterol, con 350 valores distintos, no muestra nada útil.")
    s += para("La pregunta clave es: <b>¿tiene sentido promediarla?</b> Si la respuesta es sí, es numérica. Si no, es categórica.")
    s.append(img("ap01_tipos.png", width=CONTENT_W * 0.78))
    s.append(Spacer(1, 4))
    s += para("<b>La trampa más común:</b> pandas solo sabe si una columna tiene números o texto. No sabe si es categórica. "
              "Si las regiones vienen codificadas como 1, 2 y 3, pandas las lee como números, pero siguen siendo categorías: "
              "la región 2 no es «el doble» de la región 1. <b>El tipo estadístico lo decides tú</b>, no <font name='Mono'>dtypes</font>.")
    s.append(ejemplo("una tienda online", [
        table([["Variable", "¿Promediable?", "¿Orden?", "Tipo"],
               ["Precio del producto", "Sí", "—", "Numérica continua"],
               ["N° de productos en el carrito", "Sí (solo enteros)", "—", "Numérica discreta"],
               ["Ciudad", "No", "No", "Categórica nominal"],
               ["Satisfacción (mala, regular, buena)", "No", "Sí", "Categórica ordinal"],
               ["Código postal", "No (es una etiqueta)", "No", "Categórica nominal"]],
              [62 * mm, 36 * mm, 20 * mm, CONTENT_W - 118 * mm - 18]),
    ]))
    s.append(Spacer(1, 5))
    s.append(pista(1, [P("Hazle a cada columna dos preguntas: <b>¿tiene sentido promediarla?</b> y, si no, <b>¿sus categorías "
                         "tienen un orden natural?</b> En ENS hay una sola columna con orden. Pista extra: ¿«Joven», «Adulto» y "
                         "«Adulto Mayor» se pueden ordenar?", "cell")]))

    # ------------------------------------------------------------ 2
    s.append(h("2", "Datos faltantes"))
    s += para("Un dato faltante (<font name='Mono'>NaN</font>) es una celda vacía. pandas los <b>salta en silencio</b> al "
              "calcular medias o conteos. Si no los cuentas, puedes creer que analizas a 350 personas cuando en realidad "
              "son 346.")
    s += para("La regla práctica: si faltan <b>pocos</b> (menos del 5%) y al azar, excluirlos no cambia las conclusiones. Si "
              "faltan <b>muchos</b>, hay que preguntarse <b>por qué</b> faltan antes de seguir. Por ejemplo, si en una encuesta "
              "de ingresos justo las personas de mayores ingresos no responden, excluirlas sesga el promedio hacia abajo.")
    s.append(KeepTogether([ejemplo("contar y medir vacíos", [
        code(["d.isna().sum()          # cuántos vacíos hay por columna",
              "d.isna().mean() * 100   # qué % de cada columna está vacío"]),
        Spacer(1, 3),
        P("¿Por qué <font name='Mono'>.mean()</font> da un porcentaje? <font name='Mono'>isna()</font> devuelve True o False, "
          "y Python trata True como 1 y False como 0. El promedio de unos y ceros es la <b>proporción</b> de unos.", "cell"),
    ]), Spacer(1, 5), pista(2, [P("Usa <font name='Mono'>.isna().mean()</font> sobre la columna EDAD. Luego compara el % con la "
                                  "regla del 5%.", "cell")])]))
    s.append(Spacer(1, 8))

    # ------------------------------------------------------------ 3
    s.append(h("3", "El centro: media y mediana"))
    s += para("La <b>media</b> suma todos los valores y divide por n. La <b>mediana</b> ordena los datos y toma el del medio. "
              "Si n es par, promedia los dos del centro.")
    s += para("La diferencia clave: la media <b>usa el valor exacto de cada dato</b>, así que un valor gigante la arrastra. A la "
              "mediana solo le importa la <b>posición</b>, así que un valor extremo no la mueve.")
    s.append(ejemplo("sueldos en una pyme", [
        P("Seis personas ganan 0.6, 0.7, 0.8, 0.9 y 1.0 millones, y el gerente gana 6.0.", "cell"),
        Spacer(1, 3),
        table([["Medida", "Cálculo", "Resultado"],
               ["Media", "(0.6 + 0.7 + 0.8 + 0.9 + 1.0 + 6.0) / 6", "1.67"],
               ["Mediana", "n = 6 (par) → promedio del 3° y 4° valor: (0.8 + 0.9) / 2", "0.85"]],
              [22 * mm, CONTENT_W - 62 * mm - 18, 22 * mm]),
        Spacer(1, 3),
        img("ap01_media.png", width=CONTENT_W - 22),
        P("Cinco de seis personas ganan 1 millón o menos, pero la media dice 1.67. La mediana (0.85) describe mucho "
          "mejor el sueldo típico. Por eso los ingresos se reportan con la mediana.", "cell"),
    ]))
    s.append(Spacer(1, 5))
    s.append(pista(3, [P("Antes de ejecutar, piensa: ¿cuál de las dos medidas usa el valor 50 tal como es, y cuál solo usa su "
                         "posición en la fila ordenada? Con 40 personas, agregar una más mueve la mediana como mucho medio puesto.", "cell")]))
    s.append(Spacer(1, 6))
    s.append(recuerda(["Media ≈ mediana → distribución más o menos simétrica.",
                       "Media mucho mayor que la mediana → cola de valores altos (asimetría a la derecha).",
                       "Con ingresos, precios o tiempos, que suelen tener extremos, prefiere la mediana."]))

    # ------------------------------------------------------------ 4
    s.append(h("4", "La dispersión: varianza y desviación estándar"))
    s += para("Dos cursos pueden tener la misma nota promedio y ser muy distintos: en uno todos sacan 5, en el otro la mitad saca 2 "
              "y la otra mitad 7. La <b>dispersión</b> mide cuánto se alejan los datos del centro.")
    s += para("La idea de la desviación estándar es medir la <b>distancia típica a la media</b>. El cálculo tiene 5 pasos. Los "
              "desvíos se elevan al cuadrado porque, si solo los sumaras, los positivos y negativos se cancelarían y siempre "
              "darían 0.")
    s.append(ejemplo("x = [3, 5, 7, 9]", [
        table([["Paso", "Qué haces", "Con los datos", "Resultado"],
               ["1", "Calcular la media", "(3 + 5 + 7 + 9) / 4", "6"],
               ["2", "Restar la media a cada valor (desvíos)", "3−6, 5−6, 7−6, 9−6", "−3, −1, 1, 3"],
               ["3", "Elevar cada desvío al cuadrado", "(−3)², (−1)², 1², 3²", "9, 1, 1, 9"],
               ["4", "Sumar y dividir por n − 1 → varianza", "(9 + 1 + 1 + 9) / 3", "6.67"],
               ["5", "Raíz cuadrada → desviación estándar", "√6.67", "2.58"]],
              [11 * mm, 58 * mm, 44 * mm, CONTENT_W - 113 * mm - 18]),
        Spacer(1, 3),
        P("Interpretación: los valores están, típicamente, a unas <b>2.6 unidades</b> de la media. La varianza (6.67) está en "
          "unidades al cuadrado y por eso cuesta interpretarla. Para describir datos se usa la desviación estándar.", "cell"),
    ]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>¿Por qué n − 1 y no n?</b>", "cellb"), Spacer(1, 2),
                  P("Con una muestra no conoces la media real de la población, solo la de tu muestra. Y la media de la muestra "
                    "está, por construcción, «pegada» a tus datos, así que las distancias a ella salen un poco más chicas de lo "
                    "que serían con la media real. Dividir por n − 1 en vez de n compensa ese efecto. Con muestras, siempre n − 1.", "cell"),
                  Spacer(1, 3),
                  table([["Herramienta", "Divide por", "Con x = [3, 5, 7, 9]"],
                         ["pd.Series(x).std()", "n − 1 (correcto)", "2.58"],
                         ["np.std(x, ddof=1)", "n − 1 (correcto)", "2.58"],
                         ["np.std(x)", "n (por defecto, ojo)", "2.24"]],
                        [55 * mm, 45 * mm, CONTENT_W - 100 * mm - 22])],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 5))
    s.append(pista(4, [P("Sigue los 5 pasos de la tabla con numpy: <font name='Mono'>x.mean()</font>, luego "
                         "<font name='Mono'>x - media</font>, luego <font name='Mono'>** 2</font>, luego "
                         "<font name='Mono'>.sum() / (len(x) - 1)</font> y por último <font name='Mono'>np.sqrt(...)</font>. "
                         "Después compara tu resultado con las tres herramientas de la tabla de arriba.", "cell")]))

    # ------------------------------------------------------------ 5
    s.append(h("5", "Cuartiles, IQR, boxplot y outliers"))
    s += para("Los <b>cuartiles</b> cortan los datos ordenados en 4 partes iguales. Q1 deja el 25% de los datos por debajo, "
              "la mediana (Q2) el 50% y Q3 el 75%. El <b>IQR</b> (rango intercuartílico) es Q3 − Q1: el ancho del 50% "
              "central. Como ignora los extremos, es una medida de dispersión <b>robusta</b>.")
    s += para("El <b>boxplot</b> dibuja todo eso. La caja va de Q1 a Q3, la línea es la mediana y los bigotes se extienden hasta "
              "el último dato que no se aleja más de 1.5·IQR de la caja. Lo que queda más allá se marca como punto: un "
              "<b>outlier</b>.")
    s.append(ejemplo("y = [2, 4, 5, 5, 6, 7, 8, 30]", [
        img("ap01_boxplot.png", width=CONTENT_W - 22),
        table([["Paso", "Cálculo", "Resultado"],
               ["Cuartiles (pandas)", "y.quantile([0.25, 0.5, 0.75])", "Q1 = 4.75 · Q2 = 5.5 · Q3 = 7.25"],
               ["IQR", "7.25 − 4.75", "2.5"],
               ["Límite inferior", "Q1 − 1.5·IQR = 4.75 − 3.75", "1.0"],
               ["Límite superior", "Q3 + 1.5·IQR = 7.25 + 3.75", "11.0"],
               ["Outliers", "valores < 1.0 o > 11.0", "solo el 30"]],
              [34 * mm, 62 * mm, CONTENT_W - 96 * mm - 18]),
        Spacer(1, 3),
        P("Observa que la media de y es 8.4, pero la mediana es 5.5: el 30 arrastra la media. Es lo mismo que aprendiste "
          "en la sección 3.", "cell"),
    ]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Un outlier NO es un error</b>", "cellb"),
                  P("Es un valor inusual. Puede ser un error de tipeo, pero también lo más interesante de los datos (en "
                    "detección de fraude, los outliers son justamente lo que buscas). Primero entiende por qué existe y "
                    "después decides qué hacer. La profesora insiste en esto en las Clases 17 y 18.", "cell")],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 5))
    s.append(pista(5, [P("La función tiene tres partes. Completa los espacios:", "cell"),
                       code(["def outliers(x):",
                             "    q1, q3 = x.quantile([____, ____])",
                             "    iqr = ____",
                             "    return x[(x < ____) | (x > ____)]"]),
                       P("Para aplicarla por grupo: <font name='Mono'>ens.groupby(\"EDAD\")[\"COLESTEROL\"].apply(outliers)</font>. "
                         "Para revisar el resultado, cuenta los puntos sueltos en el boxplot del notebook.", "cell")]))

    # ------------------------------------------------------------ 6
    s.append(h("6", "Comparar grupos con groupby"))
    s += para("Casi toda pregunta de análisis es una comparación: ¿los fumadores tienen más colesterol? ¿Minería paga más que Retail? "
              "En pandas eso se hace con <font name='Mono'>groupby</font>, que sigue siempre el mismo patrón: "
              "<b>separar, aplicar y combinar</b>.")
    s.append(img("ap01_groupby.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 4))
    s.append(ejemplo("la plantilla que vas a usar todo el diplomado", [
        code(["d.groupby(\"GRUPO\")[\"VARIABLE\"].agg([\"mean\", \"median\", \"std\", \"count\"])"]),
        Spacer(1, 3),
        P("Se lee: «separa por GRUPO, toma la VARIABLE y calcula estas 4 medidas en cada grupo». Incluye siempre "
          "<font name='Mono'>count</font>: una media calculada con 5 personas vale mucho menos que una calculada con 200.", "cell"),
        Spacer(1, 4),
        P("Para variables <b>categóricas</b> se cuenta en vez de promediar:", "cell"),
        code(["d[\"FUMA\"].value_counts(normalize=True)   # % de cada categoría",
              "pd.crosstab(d[\"FUMA\"], d[\"DEPORTE\"])     # tabla cruzada de dos categóricas"]),
    ]))
    s.append(Spacer(1, 5))
    s.append(pista(6, [P("Usa la plantilla dos veces: una con «FUMA» y otra con «DEPORTE» como grupo. Para la pregunta 3, "
                         "piensa qué pasaría si tomaras <b>otras</b> 350 personas: ¿saldrían exactamente las mismas medias? Esa "
                         "duda es la que resuelven las pruebas de hipótesis de la sesión 0.5.", "cell")]))
    s.append(Spacer(1, 8))

    # ------------------------------------------------------------ 7
    s.append(h("7", "Receta para el ejercicio de cierre"))
    s += para("El ejercicio de cierre junta todo. Esta es la receta que se aplica a <b>cualquier</b> análisis descriptivo, "
              "no solo a este:")
    s.append(numbered([
        ("Clasifica", "Decide el tipo de cada variable (sección 1)."),
        ("Revisa vacíos", "Usa isna() (sección 2)."),
        ("Resume por grupo", "Media, mediana, desviación estándar, IQR y n, con groupby (secciones 3 a 6)."),
        ("Grafica", "Haz un boxplot por grupo y busca outliers (sección 5)."),
        ("Concluye", "Escribe 3 frases con números y unidades."),
    ], E0))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Plantilla para escribir la conclusión</b>", "cellb"), Spacer(1, 2),
                  P("1. «<i>[Grupo A]</i> tiene el <i>[variable]</i> más alto (media <i>[x]</i>, mediana <i>[y]</i>), y "
                    "<i>[grupo B]</i> el más bajo (<i>[x]</i>).»", "cell"),
                  P("2. «La mayor dispersión está en <i>[grupo]</i> (s = <i>[x]</i>), es decir, ahí los valores son más "
                    "variados.»", "cell"),
                  P("3. «Media y mediana son <i>[parecidas / distintas]</i>, así que <i>[no hay / hay]</i> asimetría "
                    "fuerte; <i>[hay / no hay]</i> outliers.»", "cell")],
                 bg=BLUE_BG, border=BLUE))

    # ------------------------------------------------------------ cheat sheet
    s.append(h("8", "Chuleta: todo el código de la sesión"))
    s.append(table([
        ["Quiero…", "Código"],
        ["Ver las primeras filas", "d.head()"],
        ["Ver los tipos que detectó pandas", "d.dtypes"],
        ["Resumen rápido de todo", "d.describe(include=\"all\")"],
        ["Contar vacíos", "d.isna().sum()"],
        ["% de vacíos", "d.isna().mean() * 100"],
        ["Media / mediana", "d[\"X\"].mean()  ·  d[\"X\"].median()"],
        ["Desviación estándar (n − 1)", "d[\"X\"].std()"],
        ["Cuartiles", "d[\"X\"].quantile([0.25, 0.5, 0.75])"],
        ["Resumen por grupo", "d.groupby(\"G\")[\"X\"].agg([\"mean\", \"median\", \"std\", \"count\"])"],
        ["Frecuencias / %", "d[\"G\"].value_counts(normalize=True)"],
        ["Tabla cruzada", "pd.crosstab(d[\"A\"], d[\"B\"])"],
        ["Histograma", "plt.hist(d[\"X\"], bins=25)"],
        ["Boxplot por grupo", "plt.boxplot([d.loc[d.G == g, \"X\"] for g in grupos])"],
    ], [58 * mm, CONTENT_W - 58 * mm]))
    s.append(Spacer(1, 8))
    s.append(h("9", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Promediar un código (región, código postal)", "Los números son etiquetas, no cantidades", "Tratarlo como categórico"],
        ["Reportar solo la media", "Con extremos, engaña", "Reportar también la mediana"],
        ["Usar np.std() en una muestra", "Divide por n y subestima", "Usar ddof=1 o pandas"],
        ["Borrar outliers «porque sí»", "Pueden ser lo más importante", "Entender primero su origen"],
        ["Comparar medias sin mirar n", "Grupos chicos dan medias inestables", "Incluir count en el resumen"],
        ["No contar vacíos", "Analizas otro grupo del que crees", "isna().sum() al empezar"],
    ], [52 * mm, 58 * mm, CONTENT_W - 110 * mm], header_color=colors.HexColor("#DC2626")))
    s.append(Spacer(1, 8))
    s.append(recuerda(["Antes de mañana: lee estos apuntes completos (~40 min).",
                       "Mañana: abre el notebook 0.1 y resuelve los ejercicios usando solo las pistas de aquí.",
                       "Si te trabas más de 10 minutos en un ejercicio, abre la solución, entiéndela y vuelve a escribirla sin mirar."]))
    build("apuntes/0.1_Apuntes_Describir_los_datos.pdf", s, "Apuntes · Sesión 0.1 · Describir los datos")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    apuntes()
    print("OK")
