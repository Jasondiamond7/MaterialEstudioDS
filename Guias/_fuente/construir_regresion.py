"""Carpeta 1 · Regresión (Clases 17-19): apuntes de lectura y ejercicios de cuaderno."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import code, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN_BG, NAVY, OUT, P, RED, SOFT, band,
                           box, build, bullets, img, numbered, table)

C17 = colors.HexColor("#2563EB")
GREEN_C = colors.HexColor("#059669")


def h(num, title):
    return P(f'<font color="#2563EB">{num}</font>  {title}', "h2")


# ======================================================================= APUNTES
def apuntes():
    s = []
    s.append(band("Apuntes · Carpeta 1 · ANOVA y regresión lineal",
                  "Para leer antes de los ejercicios · Clases 17, 18 y 19 · lectura de ~60 min", C17))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("Un solo modelo", "Y = parte explicada + error. ANOVA explica con grupos; la regresión, con números."),
                      ("ANOVA", "Compara la variación ENTRE grupos con la variación DENTRO de ellos. F grande → los grupos difieren."),
                      ("Regresión simple", "La recta de mínimos cuadrados. β<sub>1</sub> = cambio en Y por unidad de X. R² = r²."),
                      ("Supuestos", "Normalidad, independencia y homocedasticidad de los residuos. En esas pruebas queremos p GRANDE."),
                      ("Regresión múltiple", "Cada β es el efecto de su X manteniendo las demás constantes. Categóricas con m − 1 dummies."),
                      ("Elegir modelo", "R² siempre sube: compara con R² ajustado, AIC o BIC. Vigila el VIF y no extrapoles."),
                  ], C17)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> que apunta al notebook <font name='Mono'>C17-19_Regresion.ipynb</font> o al cuaderno "
               "<font name='Mono'>02_Ejercicios_Cuaderno_Regresion.pdf</font>. La guía visual de esta carpeta es "
               "<font name='Mono'>02_Regresion_C17-19.pdf</font>.", "small"))

    # 1
    s.append(h("1", "La idea de modelo estadístico"))
    s += para("La profesora parte con el modelo más simple posible: <b>Y = μ + ε</b>. Cada dato es un promedio más un error "
              "aleatorio. Todos los modelos de esta carpeta agregan algo a ese promedio para explicar mejor Y:")
    s.append(table([["Modelo", "Fórmula", "Qué explica Y"],
                    ["Basal", "Y = μ + ε", "nada: solo el promedio"],
                    ["ANOVA (Clase 17)", "Y = μ + α<sub>grupo</sub> + ε", "el grupo (categórica)"],
                    ["Regresión simple (Clase 18)", "Y = β<sub>0</sub> + β<sub>1</sub>X + ε", "una variable numérica"],
                    ["Regresión múltiple (Clase 19)", "Y = β<sub>0</sub> + β<sub>1</sub>X<sub>1</sub> + … + β<sub>k</sub>X<sub>k</sub> + ε",
                     "varias variables, numéricas y categóricas"]],
                   [50 * mm, 70 * mm, CONTENT_W - 120 * mm]))
    s.append(Spacer(1, 4))
    s.append(recuerda(["En todos los casos, ε se supone normal, independiente y con varianza constante. Esos son los supuestos que "
                       "se revisan al final de cada modelo."]))

    # 2
    s.append(h("2", "ANOVA de un factor"))
    s += para("Pregunta típica: <b>¿el salario promedio es igual en Banca, Minería y Retail?</b> H0: todas las medias son iguales. "
              "H1: al menos dos son distintas.")
    s += para("La idea es comparar dos variaciones. La variación <b>entre</b> grupos mide cuánto se alejan las medias de cada grupo "
              "de la media general. La variación <b>dentro</b> mide cuánto se alejan los datos de la media de su propio grupo, o sea, "
              "el ruido. Si la primera es mucho mayor que la segunda, las diferencias entre grupos no son solo ruido.")
    s.append(img("reg_anova.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(ejemplo("tres grupos pequeños: A = 4, 5, 6 · B = 7, 8, 9 · C = 5, 6, 7", [
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "Medias de grupo y media general", "A 5 · B 8 · C 6 · general 6.33"],
               ["2", "SC entre = Σ n<sub>j</sub>·(media<sub>j</sub> − media general)²", "3·(1.78 + 2.78 + 0.11) = 14"],
               ["3", "SC dentro = Σ (dato − media de su grupo)²", "(1+0+1)·3 = 6"],
               ["4", "gl entre = k − 1 · gl dentro = N − k", "2 · 6"],
               ["5", "CM = SC / gl", "CM entre = 7 · CM dentro = 1"],
               ["6", "F = CM entre / CM dentro", "7"],
               ["7", "Comparar con F crítico (2, 6; α = 0.05) = 5.14", "7 &gt; 5.14 → se rechaza H0 (p = 0.027)"]],
              [11 * mm, 82 * mm, CONTENT_W - 93 * mm - 18])]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Arma la tabla en el mismo orden: medias → SC entre → SC dentro → gl → CM → F. Un control rápido: "
                      "SC entre + SC dentro = SC total, la suma de (dato − media general)² de todos los datos.", "cell")]))

    # 3
    s.append(h("3", "Después del ANOVA: comparaciones múltiples e interacción"))
    s += para("ANOVA dice <b>si</b> hay diferencias, no <b>cuáles</b>. Para eso se usa <b>Tukey</b>: compara todos los pares de grupos "
              "controlando el error tipo I. Con los salarios: Banca vs Minería y Minería vs Retail difieren (p &lt; 0.001); Banca vs "
              "Retail no (p = 0.23).")
    s += para("Con dos factores aparece la <b>interacción</b>: el efecto de uno depende del nivel del otro. En los salarios, las mujeres "
              "ganan menos en Retail y Banca, pero más en Minería. En la tabla ANOVA con interacción, Industria:Sexo tiene p = 0.033: la "
              "interacción es significativa. Gráficamente, las líneas de medias se cruzan.")
    s.append(box([P("<b>¿Por qué no hacer varias pruebas t en vez de ANOVA?</b>", "cellb"),
                  P("Con 3 grupos hay 3 pares. Si cada prueba usa α = 0.05, la probabilidad de al menos un falso positivo es "
                    "1 − 0.95³ = 14%, no 5%. ANOVA + Tukey mantienen el error total en 5%.", "cell")],
                 bg=AMBER_BG, border=AMBER))

    # 4
    s.append(h("4", "Correlación"))
    s += para("La <b>correlación de Pearson</b> (r) mide la fuerza de la relación <b>lineal</b> entre dos variables numéricas, de −1 a 1. "
              "La de <b>Spearman</b> usa rangos y detecta relaciones que siempre suben o siempre bajan, aunque no sean rectas.")
    s.append(bullets(["<b>r ≈ 0 no significa «sin relación»</b>: puede haber una curva. Siempre grafica antes (el ejemplo del datasaurus).",
                      "<b>Correlación no es causalidad</b>: dos variables pueden moverse juntas por una tercera.",
                      "La covarianza depende de las unidades; la correlación no. Por eso se compara mejor."], "body", C17))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte B del notebook",
                   [P("Calcula las dos correlaciones con <font name='Mono'>df.corr()</font> y "
                      "<font name='Mono'>df.corr(method='spearman')</font>. Si en alguna variable difieren mucho, grafícala: probablemente "
                      "la relación no es lineal o hay outliers.", "cell")]))

    # 5
    s.append(h("5", "Regresión lineal simple"))
    s += para("La recta de <b>mínimos cuadrados</b> es la que hace más pequeña la suma de los residuos al cuadrado. Sus coeficientes se "
              "calculan con dos fórmulas:")
    s.append(box([P("β<sub>1</sub> = S<sub>xy</sub> / S<sub>xx</sub> = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²", "formula"),
                  P("β<sub>0</sub> = ȳ − β<sub>1</sub>·x̄", "formula"),
                  P("r = S<sub>xy</sub> / √(S<sub>xx</sub>·S<sub>yy</sub>)  ·  R² = r² (solo en regresión simple)", "formula")],
                 bg=SOFT))
    s.append(Spacer(1, 4))
    s.append(ejemplo("x = 1, 2, 3, 4, 5 · y = 2, 4, 5, 4, 5", [
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "Medias", "x̄ = 3 · ȳ = 4"],
               ["2", "Desvíos x − x̄ y y − ȳ", "(−2, −1, 0, 1, 2) · (−2, 0, 1, 0, 1)"],
               ["3", "S<sub>xy</sub> = Σ productos de desvíos", "4 + 0 + 0 + 0 + 2 = 6"],
               ["4", "S<sub>xx</sub> = Σ (x − x̄)² · S<sub>yy</sub> = Σ (y − ȳ)²", "10 · 6"],
               ["5", "β<sub>1</sub> = 6 / 10 · β<sub>0</sub> = 4 − 0.6·3", "0.6 · 2.2"],
               ["6", "r = 6 / √(10·6) · R² = r²", "0.775 · 0.60"]]),
        Spacer(1, 3),
        img("reg_mc.png", width=CONTENT_W - 22),
        P("Lectura: cada unidad más de x sube y en 0.6, en promedio. El modelo explica el 60% de la variación de y.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>El intercepto no siempre tiene sentido</b>", "cellb"),
                  P("En la clase, BIO = −885 + 410·pH. A pH = 0 la biomasa sería −885: imposible. No es un error del modelo: pH = 0 "
                    "está muy fuera del rango observado (3.2 a 7.5). <b>Nunca interpretes ni predigas fuera del rango de X.</b>", "cell")],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte B del notebook y los Bloques 3 y 4 del cuaderno",
                   [P("En el cuaderno, sigue los 6 pasos de la tabla. En el notebook, verifica que el R² del "
                      "<font name='Mono'>summary()</font> de bio ~ ph sea igual a la correlación al cuadrado: 0.774² = 0.599.", "cell")]))

    # 6
    s.append(h("6", "Diagnóstico de residuos"))
    s += para("Un modelo puede tener buen R² y aun así no servir. Antes de confiar en sus p-valores hay que revisar los residuos. "
              "Primero se miran los gráficos y después las pruebas:")
    s.append(img("reg_residuos.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(table([["Supuesto", "Gráfico", "Prueba", "H0 de la prueba", "Queremos"],
                    ["Normalidad", "QQ-plot", "Shapiro-Wilk", "los residuos son normales", "p &gt; 0.05"],
                    ["Independencia", "residuos en orden", "Durbin-Watson", "no hay autocorrelación", "DW ≈ 2 · p &gt; 0.05"],
                    ["Homocedasticidad", "residuos vs ajustados", "Breusch-Pagan", "varianza constante", "p &gt; 0.05"]],
                   [30 * mm, 34 * mm, 28 * mm, 44 * mm, CONTENT_W - 136 * mm]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Aquí el p-valor se lee al revés</b>", "cellb"),
                  P("En estas pruebas, H0 es «el supuesto se cumple». Un p-valor GRANDE es buena noticia. Ejemplo real, bio ~ pH: "
                    "Shapiro p = 0.032 (la normalidad falla), Durbin-Watson = 0.85 (hay autocorrelación) y Breusch-Pagan p = 0.98 "
                    "(la homocedasticidad se cumple).", "cell")], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(pista("el Bloque 5 del cuaderno", [P("Para cada prueba escribe primero su H0. Luego decide: p &gt; 0.05 → no se rechaza → "
                                                   "el supuesto se cumple.", "cell")]))

    # 7
    s.append(h("7", "Regresión lineal múltiple"))
    s += para("Con varias X, cada coeficiente se interpreta <b>manteniendo constantes las demás</b>. Las variables categóricas entran "
              "como <b>dummies</b> (0/1): una categórica con m niveles usa m − 1 dummies, y la categoría omitida es la referencia.")
    s.append(ejemplo("salarios (datos reales)", [
        P("Salario = 2.004 + 0.125·Experiencia − 0.417·Mujer + 0.625·Minería − 0.322·Retail. Referencias: Hombre y Banca.", "cell"),
        Spacer(1, 3),
        table([["Coeficiente", "Lectura correcta"],
               ["0.125 (Experiencia)", "Cada año extra suma 0.125 al salario, a igual sexo e industria."],
               ["−0.417 (Mujer)", "Una mujer gana 0.417 menos que un hombre con la misma experiencia e industria."],
               ["0.625 (Minería)", "Minería paga 0.625 más que Banca (la referencia), a igual experiencia y sexo."],
               ["−0.322 (Retail)", "Retail paga 0.322 menos que Banca, pero p = 0.13: no es significativo."]],
              [40 * mm, CONTENT_W - 40 * mm - 18]),
        Spacer(1, 3),
        P("Predicción para una mujer en Minería con 10 años: 2.004 + 0.125·10 − 0.417 + 0.625 = <b>3.462</b>.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>La trampa de las dummies</b>", "cellb"),
                  P("Si incluyes las 3 dummies de industria junto al intercepto, las 3 columnas suman siempre 1, igual que la columna "
                    "del intercepto. La matriz no se puede invertir y el modelo no se puede estimar. Por eso se usa "
                    "<font name='Mono'>pd.get_dummies(..., drop_first=True)</font>.", "cell")], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte C del notebook y el Bloque 6 del cuaderno",
                   [P("Para predecir, escribe la ecuación y reemplaza: las dummies valen 1 solo si la persona está en esa categoría. "
                      "Si es de la categoría de referencia, todas sus dummies valen 0.", "cell")]))

    # 8
    s.append(h("8", "Comparar y elegir modelos"))
    s += para("El R² <b>nunca baja</b> al agregar una variable, aunque sea basura. Por eso, para comparar modelos de distinto tamaño "
              "se usan medidas que penalizan la complejidad:")
    s.append(box([P("R²<sub>ajustado</sub> = 1 − (1 − R²)·(n − 1) / (n − k − 1)   (k = n° de predictores)", "formula"),
                  P("AIC y BIC: menor es mejor. BIC penaliza más fuerte que AIC.", "formula")], bg=SOFT))
    s.append(Spacer(1, 4))
    s.append(ejemplo("¿agregar salinidad al modelo de biomasa? (n = 45)", [
        pasos([["Modelo", "R²", "R² ajustado"],
               ["bio ~ pH + BUF (k = 2)", "0.599", "1 − 0.401·44/42 = 0.580"],
               ["bio ~ pH + BUF + SAL (k = 3)", "0.604", "1 − 0.396·44/41 = 0.575"]],
              [70 * mm, 30 * mm, CONTENT_W - 100 * mm - 18]),
        Spacer(1, 3),
        P("El R² sube, pero el ajustado <b>baja</b>: SAL no aporta lo suficiente para justificar un parámetro más.", "cell")]))
    s.append(Spacer(1, 4))
    s += para("La <b>selección stepwise</b> automatiza la búsqueda. Forward parte de nada y agrega variables; backward parte de todas y "
              "quita. No prueban todas las combinaciones (con 16 variables serían 2<super>16</super> = 65.536), así que pueden terminar "
              "en modelos distintos. Con biomass, forward elige pH, type, BUF, Cu, Mn y Mg (R² = 0.85), y backward elige otro conjunto.")

    # 9
    s.append(h("9", "Multicolinealidad y VIF"))
    s += para("Si dos predictores están muy correlacionados, el modelo no logra separar el efecto de cada uno: los errores estándar se "
              "inflan y los coeficientes se vuelven inestables. El <b>VIF</b> lo mide.")
    s.append(ejemplo("calcular e interpretar el VIF", [
        P("VIF<sub>j</sub> = 1 / (1 − R²<sub>j</sub>), donde R²<sub>j</sub> es el R² de explicar X<sub>j</sub> con las demás X.", "formula"),
        Spacer(1, 2),
        pasos([["R²<sub>j</sub>", "VIF", "Lectura"],
               ["0.5", "1 / 0.5 = 2", "aceptable"],
               ["0.9", "1 / 0.1 = 10", "problema severo"],
               ["0.984", "≈ 62", "el pH en el modelo completo de biomass"]],
              [30 * mm, 40 * mm, CONTENT_W - 70 * mm - 18]),
        Spacer(1, 2),
        P("Regla práctica: VIF &lt; 5 bien · 5 a 10 revisar · &gt; 10 problema. En el modelo pH + BUF + SAL, pH y BUF tienen VIF ≈ 10: "
          "están muy correlacionados entre sí (r ≈ −0.95).", "cell")]))

    # 10
    s.append(h("10", "Predecir con datos nuevos"))
    s += para("Un modelo solo es confiable <b>dentro del rango</b> de los datos con que se ajustó. Antes de predecir, compara los rangos "
              "de las X nuevas con los de entrenamiento.")
    s.append(box([P("<b>Alerta: bio_new.csv trae un error</b>", "cellb"),
                  P("En ese archivo, Mn, Zn y Cu tienen el punto decimal corrido: Cu vale 502.381 donde en biomass vale 5.02381. Un modelo "
                    "que usa esas variables predice biomasas de ~100.000, cuando los datos van de ~200 a ~2.500. La Parte D del notebook "
                    "te pide detectarlo. Es una lección real: <b>revisa los datos antes de confiar en una predicción.</b>", "cell")],
                 bg=colors.HexColor("#FEF2F2"), border=RED))

    s.append(PageBreak())
    s.append(h("11", "Chuleta de código (statsmodels)"))
    s.append(table([
        ["Quiero…", "Código"],
        ["ANOVA de un factor", "m = smf.ols('y ~ C(g)', d).fit(); sm.stats.anova_lm(m)"],
        ["ANOVA con interacción", "smf.ols('y ~ C(a) * C(b)', d).fit()"],
        ["Tukey", "pairwise_tukeyhsd(d.y, d.g)"],
        ["Correlaciones", "d.corr() · d.corr(method='spearman') · stats.pearsonr(x, y)"],
        ["Regresión", "m = smf.ols('y ~ x1 + x2 + C(cat)', d).fit(); m.summary()"],
        ["R², R² ajustado, AIC", "m.rsquared · m.rsquared_adj · m.aic · m.bic"],
        ["Normalidad", "stats.shapiro(m.resid) · sm.qqplot(m.resid, line='s')"],
        ["Independencia", "durbin_watson(m.resid)"],
        ["Homocedasticidad", "het_breuschpagan(m.resid, m.model.exog)"],
        ["Puntos influyentes", "m.get_influence().cooks_distance[0]"],
        ["VIF", "variance_inflation_factor(X.values, j)"],
        ["Predecir con intervalo", "m.get_prediction(nuevos).summary_frame(alpha=0.05)"],
    ], [44 * mm, CONTENT_W - 44 * mm]))
    s.append(Spacer(1, 8))
    s.append(h("12", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Hacer varias pruebas t con 3+ grupos", "Infla el error tipo I", "ANOVA + Tukey"],
        ["Comparar modelos con R²", "R² siempre sube", "R² ajustado, AIC, BIC"],
        ["Interpretar β sin «a igual lo demás»", "En la múltiple cada β es parcial", "Decir siempre qué se mantiene fijo"],
        ["Incluir todas las dummies", "Matriz no invertible", "m − 1 dummies (drop_first)"],
        ["Leer p &lt; 0.05 en Shapiro como bueno", "En supuestos, H0 = se cumple", "Queremos p grande"],
        ["Predecir fuera del rango de X", "El modelo no sabe qué pasa ahí", "Comparar rangos antes"],
        ["Borrar outliers por receta", "Pueden ser lo más importante", "Entender su origen primero"],
    ], [52 * mm, 50 * mm, CONTENT_W - 102 * mm], header_color=RED))
    s.append(Spacer(1, 8))
    s.append(recuerda(["Orden sugerido: lee estos apuntes → Bloques 1–2 del cuaderno → Parte A del notebook → Bloques 3–5 → "
                       "Parte B → Bloques 6–8 → Partes C y D → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/02_Apuntes_Regresion_C17-19.pdf", s, "Apuntes · Carpeta 1 · ANOVA y regresión · Clases 17–19")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#2563EB">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow.append(P(intro, "cell"))
        flow.append(Spacer(1, 3))
    flow.append(numbered([("", t) for t in items], C17))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Carpeta 1 · ANOVA y regresión",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C17))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo, no solo el resultado.",
                           "Redondea a 2 o 3 decimales. Pequeñas diferencias de redondeo con las soluciones son normales.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · "
                           "<font name='Sym'>★★★</font> integra varios conceptos.",
                           "Corrige con las soluciones y marca con color lo que tuviste mal: eso va al repaso del domingo."],
                          "cell", C17)], bg=SOFT, border=C17))
    s.append(Spacer(1, 4))
    s += bloque(1, "ANOVA a mano", "★★", "Tres métodos de estudio, puntaje de 3 estudiantes cada uno: "
                "M1 = 6, 7, 8 · M2 = 9, 10, 11 · M3 = 6, 8, 10.", [
        "Calcula la media de cada grupo y la media general.",
        "Calcula la SC entre, la SC dentro y verifica que suman la SC total.",
        "Completa la tabla ANOVA: gl, CM y F.",
        "Con F crítico (2, 6; α = 0.05) = 5.14, ¿rechazas H0? Escribe H0, H1 y la conclusión en palabras.",
    ])
    s += bloque(2, "Leer tablas ANOVA (datos reales de salarios)", "★★", None, [
        "En Salario ~ Industria: SC Industria = 17.88 con 2 gl; SC residual = 18.04 con 37 gl. Calcula los CM y el F. ¿Cuántas personas "
        "hay en total?",
        "Con el modelo con interacción, Industria:Sexo tiene F = 3.77 y p = 0.033. ¿Qué significa en palabras sobre hombres y mujeres "
        "en las distintas industrias?",
        "Tukey da: Banca–Minería p = 0.0008 · Banca–Retail p = 0.23 · Minería–Retail p &lt; 0.001. ¿Qué pares difieren? Resume en una frase.",
        "<font name='Sym'>★★★</font> Si hicieras 3 pruebas t independientes con α = 0.05, ¿cuál sería la probabilidad de al menos un "
        "falso positivo? ¿Qué te dice eso sobre usar ANOVA?",
    ])
    s += bloque(3, "Correlación y regresión simple a mano", "★★", "Horas de estudio (x) y nota (y): x = 2, 4, 6, 8 · y = 3, 5, 4, 8.", [
        "Calcula x̄, ȳ, S<sub>xy</sub>, S<sub>xx</sub> y S<sub>yy</sub>.",
        "Calcula β<sub>1</sub>, β<sub>0</sub> y escribe la recta.",
        "Calcula r y R². Interpreta R² en una frase.",
        "Calcula los 4 residuos y verifica que suman 0.",
        "Predice la nota para 5 horas. ¿Sería razonable predecir para 20 horas? ¿Por qué?",
    ])
    s += bloque(4, "Interpretar bio ~ pH (datos reales)", "★", "Modelo de clase: BIO = −885.2 + 409.8 · pH, R² = 0.599.", [
        "Predice la biomasa para pH = 4 y pH = 5. ¿Cuánto cambia?",
        "Interpreta la pendiente en una frase con unidades.",
        "¿Tiene sentido el intercepto? Justifica (en los datos, el pH va de 3.2 a 7.5).",
        "¿Cuánto vale r? ¿Qué signo tiene y por qué?",
    ])
    s += bloque(5, "Diagnóstico de residuos", "★★", "Resultados de bio ~ pH: Shapiro p = 0.032 · Durbin-Watson = 0.85 · Breusch-Pagan p = 0.98.", [
        "Para cada prueba, escribe su H0 y decide si el supuesto se cumple.",
        "Los datos de biomass están ordenados por localidad y tipo de vegetación. ¿Cómo podría explicar eso el resultado de Durbin-Watson?",
        "<font name='Sym'>★★★</font> Un compañero dice: «Shapiro dio p &lt; 0.05, así que el resultado es significativo y el modelo es "
        "bueno». Explica su error.",
    ])
    s += bloque(6, "Regresión múltiple con dummies (salarios)", "★★",
                "Salario = 2.004 + 0.125·Exp − 0.417·Mujer + 0.625·Minería − 0.322·Retail. Referencias: Hombre y Banca.", [
        "Predice el salario de: (a) hombre en Banca con 5 años; (b) mujer en Retail con 5 años; (c) mujer en Minería con 10 años.",
        "Interpreta el coeficiente −0.417 con la frase «manteniendo constante…».",
        "¿Por qué el modelo tiene solo 2 dummies de industria y no 3?",
        "<font name='Sym'>★★★</font> Retail tiene p = 0.13. ¿Significa que Retail paga igual que Banca? Responde con cuidado.",
    ])
    s += bloque(7, "R² ajustado y VIF", "★★", None, [
        "Con n = 45: el modelo A (k = 2) tiene R² = 0.599 y el modelo B (k = 3) tiene R² = 0.604. Calcula ambos R² ajustados. ¿Cuál "
        "prefieres?",
        "Calcula el VIF si R²<sub>j</sub> = 0.75 y si R²<sub>j</sub> = 0.95. Clasifica cada uno.",
        "En el modelo completo de biomass, el pH tiene VIF = 62. ¿Qué R²<sub>j</sub> implica? ¿Qué le pasa a su coeficiente?",
    ])
    s += bloque(8, "Selección de modelos y predicción", "★★★", None, [
        "¿Cuántos subconjuntos posibles de variables hay con 5 candidatas? ¿Y con 16?",
        "Forward eligió pH, type, BUF, Cu, Mn, Mg (AIC 642.1) y backward eligió loc, type, Ca, Mg, Mn, Cu, NH4 (AIC 639.3). ¿Por qué "
        "pueden diferir? ¿Cuál elegirías por AIC?",
        "En bio_new.csv, Cu vale 502.381 donde en biomass vale 5.02381. ¿Qué pasará si predices con un modelo que usa Cu? ¿Qué harías?",
    ])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "M1 = 7 · M2 = 10 · M3 = 8 · media general = 75 / 9 = 8.33."],
        ["2", "SC entre = 3·[(7 − 8.33)² + (10 − 8.33)² + (8 − 8.33)²] = 3·(1.78 + 2.78 + 0.11) = 14. SC dentro = (1+0+1) + (1+0+1) + "
              "(4+0+4) = 12. SC total = Σ(dato − 8.33)² = 26 = 14 + 12."],
        ["3", "gl entre = 2, gl dentro = 6. CM entre = 7, CM dentro = 2. F = 3.5."],
        ["4", "H0: μ1 = μ2 = μ3. H1: al menos dos difieren. F = 3.5 &lt; 5.14 → no se rechaza H0: con estos datos no hay evidencia de "
              "que los métodos difieran (p ≈ 0.10). Ojo: M3 tiene mucha variación interna, y eso «tapa» la diferencia."]])
    s += sol("Bloque 2", [
        ["1", "CM Industria = 17.88 / 2 = 8.94 · CM residual = 18.04 / 37 = 0.488 · F = 8.94 / 0.488 = 18.3. N = 2 + 37 + 1 = 40."],
        ["2", "La brecha entre hombres y mujeres no es igual en todas las industrias: depende de la industria (en Minería incluso se invierte)."],
        ["3", "Minería difiere de Banca y de Retail; Banca y Retail no difieren entre sí. Minería paga más que las otras dos."],
        ["4", "1 − 0.95³ = 0.143 → 14%, casi el triple del 5% buscado. ANOVA + Tukey controla el error total."]])
    s += sol("Bloque 3", [
        ["1", "x̄ = 5 · ȳ = 5. Desvíos x: −3, −1, 1, 3 · desvíos y: −2, 0, −1, 3. S<sub>xy</sub> = 6 + 0 − 1 + 9 = 14 · "
              "S<sub>xx</sub> = 20 · S<sub>yy</sub> = 14."],
        ["2", "β<sub>1</sub> = 14 / 20 = 0.7 · β<sub>0</sub> = 5 − 0.7·5 = 1.5 → ŷ = 1.5 + 0.7x."],
        ["3", "r = 14 / √(20·14) = 0.837 · R² = 0.70: las horas de estudio explican el 70% de la variación de la nota."],
        ["4", "ŷ = 2.9, 4.3, 5.7, 7.1 → residuos 0.1, 0.7, −1.7, 0.9. Suma = 0."],
        ["5", "ŷ(5) = 5.0. Para 20 horas no: está muy fuera del rango observado (2 a 8) y la relación podría no seguir siendo lineal."]])
    s += sol("Bloque 4", [
        ["1", "pH 4 → −885.2 + 1639.2 = 754 · pH 5 → 1163.8. Cambia 409.8."],
        ["2", "Por cada unidad más de pH, la biomasa aumenta en promedio 409.8 unidades."],
        ["3", "No: pH = 0 está fuera del rango observado. El intercepto solo «ancla» la recta; aquí no tiene interpretación física."],
        ["4", "r = √0.599 = 0.774, positivo porque la pendiente es positiva."]])
    s.append(PageBreak())
    s += sol("Bloque 5", [
        ["1", "Shapiro, H0: residuos normales → p = 0.032 &lt; 0.05 → se rechaza: la normalidad falla. Durbin-Watson, H0: no hay "
              "autocorrelación → 0.85, lejos de 2 → hay autocorrelación positiva. Breusch-Pagan, H0: varianza constante → p = 0.98 → "
              "se cumple."],
        ["2", "Filas vecinas son de la misma localidad y tipo, y se parecen más entre sí: sus residuos quedan correlacionados."],
        ["3", "En una prueba de supuestos, H0 = «el supuesto se cumple». p &lt; 0.05 significa que se rechaza la normalidad: es una mala noticia."]])
    s += sol("Bloque 6", [
        ["1", "(a) 2.004 + 0.625 = 2.629 · (b) 2.004 + 0.625 − 0.417 − 0.322 = 1.890 · (c) 2.004 + 1.25 − 0.417 + 0.625 = 3.462."],
        ["2", "Manteniendo constantes la experiencia y la industria, una mujer gana en promedio 0.417 menos que un hombre."],
        ["3", "Con 3 categorías se usan 3 − 1 = 2 dummies. Banca es la referencia (sus dos dummies valen 0). Usar 3 dummies + intercepto "
              "hace la matriz no invertible."],
        ["4", "No. Significa que no hay evidencia suficiente para afirmar que difieren. «No rechazar H0» no es lo mismo que «H0 es cierta»."]])
    s += sol("Bloque 7", [
        ["1", "A: 1 − 0.401·44/42 = 0.580 · B: 1 − 0.396·44/41 = 0.575. Prefiero A: el ajustado baja al agregar la tercera variable."],
        ["2", "1 / 0.25 = 4 (aceptable) · 1 / 0.05 = 20 (problema severo)."],
        ["3", "R²<sub>j</sub> = 1 − 1/62 = 0.984: el pH se explica casi por completo con las otras variables. Su coeficiente es inestable "
              "y su error estándar está muy inflado."]])
    s += sol("Bloque 8", [
        ["1", "2<super>5</super> = 32 · 2<super>16</super> = 65.536."],
        ["2", "Cada método avanza paso a paso y se detiene cuando ningún cambio mejora el AIC: no prueba todas las combinaciones. Por AIC, "
              "backward (639.3 &lt; 642.1), pero hay que revisar sus supuestos y su VIF antes de quedarse con él."],
        ["3", "La predicción explota (valores de ~100.000), porque Cu es 100 veces mayor que cualquier valor de entrenamiento. Reportaría "
              "el error, corregiría los datos o usaría un modelo sin esas variables (por ejemplo, bio ~ pH + type)."]])
    build("apuntes/02_Ejercicios_Cuaderno_Regresion.pdf", s, "Ejercicios para el cuaderno · Carpeta 1 · ANOVA y regresión")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    apuntes()
    cuaderno()
    print("OK")
