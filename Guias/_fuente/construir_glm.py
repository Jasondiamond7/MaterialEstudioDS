"""Carpeta 2 · GLM (Clases 20-21): guía visual, apuntes de lectura y ejercicios de cuaderno."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, class_page, img,
                           numbered, table, two_cols)

C20 = colors.HexColor("#0891B2")
BLUE = colors.HexColor("#2563EB")
BLUE_BG = colors.HexColor("#EFF6FF")
GREEN_C = colors.HexColor("#059669")
PURPLE = colors.HexColor("#7C3AED")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Carpeta 2 · Modelos lineales generalizados (GLM)",
                  "Clases 20 y 21 · Profesora Maritza Márquez · Etapa 2 del roadmap (semanas 6–7)", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("La gran idea de la carpeta", "cellb"), Spacer(1, 3),
                  P("Un GLM es una regresión que <b>se adapta al tipo de Y</b>. Se arma con tres piezas: "
                    "una <b>distribución</b> para Y, la misma <b>combinación lineal</b> de siempre "
                    "(β<sub>0</sub> + β<sub>1</sub>x + …) y una <b>función de enlace</b> que conecta ambas. "
                    "La logística, la Poisson y la multinomial salen del mismo molde: solo cambian las piezas.", "body")],
                 bg=colors.HexColor("#ECFEFF"), border=C20))
    s.append(Spacer(1, 6))
    s.append(P("Mapa conceptual", "h2"))
    s.append(img("glm_mapa.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 4))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Regresión lineal y lectura de coeficientes (Clases 17–19)",
                     "p-valor, prueba de hipótesis e intervalos (Etapa 0)",
                     "Logaritmo y exponencial: log(e<super>x</super>) = x",
                     "Tipos de variables (sesión 0.1)"], "cell", C20)]
    right = [P("Hacia dónde lleva", "h3"),
             bullets(["<b>A/B testing (C22):</b> comparar proporciones y odds",
                      "<b>Clasificación en ML (C30+):</b> la logística es la base de muchos clasificadores",
                      "<b>Naive Bayes (C40):</b> otra forma de estimar probabilidades de clase",
                      "<b>Datos desbalanceados:</b> la trampa de la exactitud vuelve a aparecer"], "cell", C20)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 5))
    s.append(table([["Dataset (en Guias/datos/)", "Modelo", "Pregunta"],
                    ["wcgs.csv · 3154 hombres", "Logística", "¿Fumar aumenta el riesgo de enfermedad coronaria (chd)?"],
                    ["cancer_pulmon.csv · 24 filas", "Poisson + offset", "¿Cómo cambia la tasa de cáncer según edad y ciudad?"],
                    ["nes96.csv · 944 votantes", "Multinomial", "¿Edad y educación predicen la afiliación política?"]],
                   [50 * mm, 30 * mm, CONTENT_W - 80 * mm]))
    s.append(P("En clase, wcgs y nes96 se cargan desde el paquete de R <i>faraway</i>, y los datos de cáncer se escriben "
               "dentro del notebook. Para que puedas trabajar en Python, los dejé como CSV.", "small"))
    s.append(PageBreak())

    class_page(
        s, C20, 20, "La idea de los GLM",
        "Cuando Y es un sí/no, un conteo o una categoría, la recta de siempre falla. El GLM conserva la "
        "combinación lineal, pero la conecta con Y a través de una función de enlace.",
        [("Por qué falla la regresión lineal", "Con Y binaria, la recta predice «probabilidades» menores que 0 o mayores que 1. Con conteos, la varianza crece con la media."),
         ("Componente 1 · Distribución", "La elige el TIPO de Y: Binomial (sí/no), Poisson (conteos), Multinomial (3+ categorías), Normal (continua)."),
         ("Componente 2 · Predictor lineal", "η = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + … : la misma combinación lineal de la regresión."),
         ("Componente 3 · Enlace g(μ) = η", "Transforma la media de Y para que viva en (−∞, +∞): logit para probabilidades, log para conteos, identidad para la normal."),
         ("Odds y logit", "odds = p / (1 − p). logit = log(odds). La logística modela el logit como una recta.")],
        ["η = β<sub>0</sub> + β<sub>1</sub>x",
         "logit(p) = log(p / (1 − p)) = η",
         "p = 1 / (1 + e<super>−η</super>)",
         "Poisson: log(μ) = η  →  μ = e<super>η</super>"],
        ["<b>Estimate</b>: en escala del enlace (log-odds o log), NO en la escala de Y",
         "<b>z value / Pr(&gt;|z|)</b>: prueba H0: β = 0",
         "<b>Null vs Residual deviance</b>: cuánto mejora el ajuste al agregar las X",
         "<b>AIC</b>: comparar modelos (menor es mejor)"],
        "glm_porque.png",
        "Izquierda: datos ilustrativos; la recta se sale del rango posible y la curva S no. Derecha: datos reales de wcgs. "
        "Los círculos son el % real por grupo de cigarrillos y la curva es el modelo logístico.",
        ["La distribución se elige según el tipo de Y. El enlace se elige después, entre los disponibles para esa "
         "distribución (logit, probit, cloglog para binomial). «El software aguanta lo que le pongas»: justifícalo.",
         "El intercepto rara vez tiene interpretación práctica."],
        ["import statsmodels.api as sm",
         "import statsmodels.formula.api as smf",
         "smf.glm('y ~ x', d,",
         "    family=sm.families.Binomial()).fit()"])

    class_page(
        s, BLUE, 21, "Regresión logística (wcgs)",
        "Cada coeficiente dice cuánto cambian las log-odds. Al exponenciarlo, e<super>β</super> dice por cuánto "
        "se MULTIPLICAN las odds: ese es el odds ratio (OR).",
        [("El modelo", "log(p / (1 − p)) = −2.742 + 0.0232·cigs. Y = chd (enfermedad coronaria, sí/no); el 8.1% de los 3154 hombres la tiene."),
         ("Odds ratio", "OR = e<super>0.0232</super> = 1.0235: cada cigarrillo diario adicional multiplica las odds por 1.0235 (+2.35%). Por 10 cigarrillos: e<super>10·β</super> = 1.26 (+26%)."),
         ("Signo de β", "β &gt; 0 → OR &gt; 1 (factor de riesgo). β = 0 → OR = 1 (sin efecto). β &lt; 0 → OR &lt; 1 (factor protector)."),
         ("Probabilidad predicha", "Para 0 cigarrillos: p = 1/(1 + e<super>2.742</super>) = 0.061. Para 20: 0.093. Para 40: 0.140."),
         ("Comparar modelos", "Agregar edad baja la deviance de 1750.0 a 1705.9. Prueba de razón de verosimilitud: χ² = 44.1, p = 3·10<super>−11</super> → la edad aporta.")],
        ["OR = e<super>β</super>   ·   OR por k unidades = e<super>k·β</super>",
         "% de cambio en odds = (OR − 1) · 100",
         "LR = deviance<sub>simple</sub> − deviance<sub>completo</sub> ~ χ²",
         "gl = n° de parámetros agregados"],
        ["Intercepto −2.742 · cigs 0.0232 (EE 0.004, p ≈ 9·10<super>−9</super>)",
         "IC 95% del OR de cigs: [1.015; 1.032]: no incluye 1",
         "AIC: logit 1753.97 · probit 1753.49 (casi iguales)",
         "La curva de la figura sube con cigarrillos Y con edad"],
        "glm_logistica.png",
        "Modelo chd ~ cigs + age ajustado con wcgs. A igual número de cigarrillos, la probabilidad es mucho mayor a los 59 años "
        "que a los 40: OR por 10 años de edad = e<super>10·0.076</super> = 2.15.",
        ["Odds NO es probabilidad. «Las odds aumentan 26%» no significa que la probabilidad aumente 26 puntos.",
         "Alerta del material: un OR de 1.2614 es +26%, no +35% (ver página de alertas)."],
        ["m = smf.logit('chd01 ~ cigs', w).fit()",
         "np.exp(m.params)        # odds ratios",
         "np.exp(m.conf_int())    # IC de los OR",
         "m.predict(pd.DataFrame({'cigs': [0, 20]}))"])

    class_page(
        s, BLUE, "21", "Evaluar un clasificador",
        "Un modelo logístico entrega probabilidades. Para clasificar hay que elegir un umbral, y una exactitud "
        "alta puede esconder un modelo que no detecta a nadie.",
        [("Matriz de confusión", "Cruza lo real con lo predicho: VP, FP, FN, VN. Todo lo demás se calcula desde ahí."),
         ("La trampa de la exactitud", "En clase, con umbral 0.5: exactitud 93% pero sensibilidad 0%. El modelo dice «no» a todos y acierta porque solo el 8% está enfermo."),
         ("Sensibilidad y especificidad", "Sensibilidad: de los enfermos, ¿a cuántos detecto? Especificidad: de los sanos, ¿a cuántos descarto? Precisión: de los que marco, ¿cuántos son enfermos?"),
         ("El umbral 0.5 no es una ley", "Es «una costumbre». Bajarlo detecta más enfermos a costa de más falsas alarmas. El índice de Youden busca el mejor equilibrio."),
         ("ROC y AUC", "La curva ROC muestra todos los umbrales a la vez. AUC = 0.5 es azar; 1 es perfecto. En clase dio ≈ 0.60: un modelo débil.")],
        ["Exactitud = (VP + VN) / total",
         "Sensibilidad = VP / (VP + FN)",
         "Especificidad = VN / (VN + FP)",
         "Precisión = VP / (VP + FP)",
         "Youden = sensibilidad + especificidad − 1"],
        ["Umbral 0.5: nadie es predicho como enfermo",
         "Umbral de Youden (clase: 0.069): sens. 61%, esp. 55%",
         "Siempre reporta sensibilidad y especificidad, no solo exactitud",
         "Con clases desbalanceadas, mira el AUC"],
        "glm_clasif.png",
        "Modelo chd ~ cigs + age evaluado sobre todos los datos de wcgs (en clase se usó un 30% de prueba y solo cigs, por eso "
        "los números difieren un poco). Con umbral 0.5 nadie se clasifica como enfermo; con el de Youden se detecta al 79%.",
        ["Evalúa SIEMPRE en datos que el modelo no vio (train/test). Evaluar sobre los mismos datos de entrenamiento "
         "sobreestima el rendimiento.",
         "Elegir el umbral es una decisión de negocio: ¿qué cuesta más, un falso negativo o un falso positivo?"],
        ["from sklearn.metrics import (",
         "    confusion_matrix, roc_auc_score, roc_curve)",
         "pred = (prob >= 0.5).astype(int)",
         "confusion_matrix(y_test, pred)"])

    class_page(
        s, GREEN_C, "21", "Regresión de Poisson (conteos y tasas)",
        "Para conteos se modela el logaritmo de la media. Con un offset se modela una TASA (casos por persona), "
        "y e<super>β</super> se lee como razón de tasas (RR).",
        [("Conteos", "Y = 0, 1, 2, … sin tope. En la Poisson, media = varianza = λ. Enlace log: log(μ) = η."),
         ("Offset", "Ciudades con más población tienen más casos aunque el riesgo sea igual. log(población) entra al modelo con coeficiente fijo 1 y convierte el conteo en tasa."),
         ("Razón de tasas", "e<super>β</super>: Kolding RR = 0.69 → su tasa es 31% menor que la de Fredericia (referencia). Edad 70–74: RR = 6.4 → tasa 6.4 veces la de 40–54."),
         ("Sobredispersión", "φ = X² de Pearson / gl = 22.56 / 15 = 1.50. Si φ ≈ 1, la Poisson va bien; si φ &gt; 1, los errores estándar están subestimados."),
         ("Quasi-Poisson", "Mismos coeficientes, errores estándar inflados por √φ. En clase, Kolding deja de ser significativa al 5%.")],
        ["log(μ) = log(población) + β<sub>0</sub> + β<sub>1</sub>x",
         "tasa = μ / población = e<super>β0 + β1x</super>",
         "RR = e<super>β</super>",
         "φ = X²<sub>Pearson</sub> / (n − p)"],
        ["Deviance 129.9 (nulo) → 23.4 (con edad y ciudad), gl = 15",
         "Edad aporta mucho: Δdeviance = 101.6, p ≈ 0",
         "Ciudad no aporta una vez incluida la edad: Δdeviance = 4.86, p = 0.18",
         "φ = 1.50 → sobredispersión leve"],
        "glm_poisson.png",
        "Tasas reales por 1000 habitantes. El patrón de edad es claro; las diferencias entre ciudades son ruidosas y "
        "se cruzan. Por eso «ciudad» deja de ser significativa al controlar por edad.",
        ["Sin offset estarías comparando conteos, no riesgos: una ciudad grande «parecería» más peligrosa solo por tener más gente."],
        ["smf.glm('casos ~ C(ciudad) + C(edad)', c,",
         "    family=sm.families.Poisson(),",
         "    offset=np.log(c.poblacion)).fit()",
         "m.pearson_chi2 / m.df_resid   # φ",
         ".fit(scale='X2')              # quasi-Poisson"])

    class_page(
        s, PURPLE, "21", "Regresión multinomial (3+ categorías)",
        "Con J categorías se elige una de referencia y se ajustan J − 1 ecuaciones logit, una por cada categoría "
        "contra la referencia. Las probabilidades resultantes suman 1.",
        [("Categoría de referencia", "En clase: Demócrata. Cada ecuación compara Republicano vs Demócrata e Independiente vs Demócrata."),
         ("Coeficientes", "e<super>β</super> es una razón de riesgos relativos: cuánto se multiplica P(categoría)/P(referencia) por unidad de x."),
         ("Probabilidades", "P(j) = e<super>ηj</super> / (1 + Σ e<super>ηk</super>), con η = 0 para la referencia. Siempre suman 1."),
         ("La clase minoritaria", "Independiente es el 4% de los datos: el modelo nunca la elige como clase más probable. Exactitud 57.6%."),
         ("Nominal vs ordinal", "Nominal: sin orden (partido político). Ordinal: con orden (malo/regular/bueno) → logit acumulativo. R usa contrastes polinomiales en factores ordenados: la profesora los cambia a contr.treatment.")],
        ["log(P(j) / P(ref)) = α<sub>j</sub> + β<sub>j</sub>x,  j = 1 … J−1",
         "P(j) = e<super>ηj</super> / (1 + Σ<sub>k</sub> e<super>ηk</super>)",
         "Logit acumulativo: logit P(Y ≤ j) = α<sub>j</sub> − βx"],
        ["Modelo de clase (edad + educ + ingreso): deviance 1458.5, AIC 1582.5",
         "vs nulo: LR = 105.8 con 60 gl, p = 0.0002 → el modelo aporta",
         "1er individuo: D 0.69 · R 0.24 · I 0.07",
         "Exactitud 57.6%; Independiente nunca predicho"],
        "glm_multinom.png",
        "Modelo simplificado edad + educación (numérica) con los datos reales de nes96, para una persona de 45 años. "
        "Con más educación sube la probabilidad de ser Republicano.",
        ["Una exactitud de 57.6% suena baja, pero el modelo nulo (decir siempre «Demócrata») acierta 51.7%. Compara siempre contra esa base."],
        ["n['y'] = n.PID3.map({'Democrat': 0,",
         "    'Republican': 1, 'Independent': 2})",
         "m = smf.mnlogit('y ~ age + educ_num', n).fit()",
         "np.exp(m.params)   # razones de riesgo",
         "m.predict(nuevos)  # 3 probabilidades"])

    # --------------------------------------------------------- R -> Python
    s.append(band("Traductor R → Python y decisión de modelo", "Para leer los notebooks de clase y practicar en Python", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["En la clase (R)", "En Python (statsmodels / sklearn)"],
                    ["glm(y ~ x, family = binomial)", "smf.glm('y ~ x', d, family=sm.families.Binomial()).fit()  ·  smf.logit(...)"],
                    ["binomial(link = 'probit')", "sm.families.Binomial(link=sm.families.links.Probit())"],
                    ["glm(..., family = poisson) + offset(log(pob))", "smf.glm(..., family=sm.families.Poisson(), offset=np.log(d.pob))"],
                    ["family = quasipoisson", ".fit(scale='X2')"],
                    ["multinom(y ~ x)  (nnet)", "smf.mnlogit('y ~ x', d).fit()  (y codificada 0, 1, 2; 0 = referencia)"],
                    ["exp(coef(m)) · confint(m)", "np.exp(m.params) · np.exp(m.conf_int())"],
                    ["anova(m1, m2, test = 'Chisq')", "LR = m1.deviance − m2.deviance; stats.chi2.sf(LR, gl)"],
                    ["predict(m, type = 'response')", "m.predict(nuevos)"],
                    ["pROC::roc · caret::confusionMatrix", "sklearn.metrics: roc_curve, roc_auc_score, confusion_matrix"]],
                   [62 * mm, CONTENT_W - 62 * mm]))
    s.append(Spacer(1, 10))
    s.append(P("¿Qué modelo uso?", "h2"))
    s.append(table([["Variable respuesta Y", "Ejemplo", "Distribución", "Enlace", "Se interpreta con"],
                    ["Continua", "peso, salario", "Normal", "identidad", "β (unidades de Y)"],
                    ["Binaria (sí/no)", "¿compra?, ¿enferma?", "Binomial", "logit", "OR = e<super>β</super>"],
                    ["Conteo", "n° de reclamos, casos", "Poisson", "log", "RR = e<super>β</super>"],
                    ["Conteo con exposición", "casos por habitante", "Poisson + offset", "log", "razón de tasas e<super>β</super>"],
                    ["3+ categorías sin orden", "partido, transporte", "Multinomial", "logit generalizado", "e<super>β</super> vs referencia"],
                    ["3+ categorías con orden", "malo/regular/bueno", "Multinomial ordinal", "logit acumulativo", "OR acumulado"]],
                   [36 * mm, 35 * mm, 30 * mm, 29 * mm, CONTENT_W - 130 * mm]))
    s.append(PageBreak())

    # --------------------------------------------------------- alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Lo que hay que corregir, lo que debes saber responder y cuándo pasar a la Clase 22", NAVY))
    s.append(Spacer(1, 8))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Error en la interpretación del odds ratio (Clase21_R.Logistica, sección de resumen de los odds)</b>", "cellb"),
                  Spacer(1, 2),
                  P("El texto calcula bien OR<sub>10</sub> = e<super>10·0.02322</super> = 1.2614, pero en la frase siguiente dice que eso significa "
                    "«odds 1.35 veces más… aproximadamente un 35% mayores».", "cell"),
                  P("<b>Correcto:</b> 1.2614 significa odds <b>1.26 veces</b>, es decir, <b>26% mayores</b>. El «35%» parece un residuo de una "
                    "versión anterior del ejemplo. La tabla que viene justo después sí muestra el valor correcto.", "cell")],
                 bg=RED_BG, border=RED))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Otras observaciones</b>", "cellb"),
                  bullets(["La descripción de wcgs llama «sbp» a la presión sistólica, pero en los datos la columna se llama <b>sdp</b>.",
                           "La deviance se explica de forma intuitiva («falta de ajuste»). Su definición formal es "
                           "D = −2·(logL<sub>modelo</sub> − logL<sub>saturado</sub>), y por eso las diferencias de deviance siguen una χ².",
                           "El p-valor se usa sin definirlo. Repasa la sesión 0.5 y la alerta de la Clase 17.",
                           "El ejemplo de exactitud 93% con sensibilidad 0% no es un error: es una excelente lección. Recuérdalo."],
                          "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 8))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    preg = [("¿Por qué no usar regresión lineal para una Y de sí/no?",
             "Predice valores fuera de [0, 1] y la varianza de una proporción no es constante."),
            ("¿Qué decide la distribución de un GLM, y qué decide el enlace?",
             "La distribución, el tipo de Y. El enlace se elige entre los compatibles con esa distribución."),
            ("β = 0.0232 en una logística. ¿Qué significa e<super>β</super> = 1.0235?",
             "Cada unidad extra de x multiplica las odds por 1.0235, es decir, +2.35%."),
            ("Un modelo tiene exactitud 93%. ¿Es bueno?",
             "Depende: si el 92% es de una clase, decir siempre «no» ya da 92%. Mira sensibilidad, especificidad y AUC."),
            ("¿Para qué sirve el offset en una Poisson?",
             "Para modelar tasas: ajusta por la exposición (población, tiempo) y así se comparan riesgos, no conteos."),
            ("φ = 1.5 en una Poisson. ¿Qué haces?",
             "Hay sobredispersión: usa quasi-Poisson (o binomial negativa) para corregir los errores estándar.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 78 * mm, CONTENT_W - 85 * mm]))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 22</b>", "cellb"), Spacer(1, 3),
                  bullets(["Elijo distribución y enlace para cualquier tipo de Y",
                           "Convierto a mano entre probabilidad, odds y logit",
                           "Interpreto un OR y un RR en una frase, con %",
                           "Calculo sensibilidad y especificidad desde una matriz de confusión",
                           "Ajusto en Python una logística, una Poisson con offset y una multinomial",
                           "Resuelvo los bloques 1 a 6 del cuaderno con al menos 80% de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("03_GLM_C20-21.pdf", s, "Carpeta 2 · Modelos lineales generalizados · Clases 20–21")


# ======================================================================= APUNTES
E0 = colors.HexColor("#475569")


def h(num, title):
    return P(f'<font color="#0891B2">{num}</font>  {title}', "h2")


def para(text):
    return [P(text), Spacer(1, 4)]


def ejemplo(title, flow):
    return box([P(f"<b>Ejemplo resuelto · {title}</b>", "cellb"), Spacer(1, 3)] + flow, bg=SOFT, border=E0)


def pista(where, flow):
    return box([P(f"<b>Pista para {where}</b>", "cellb"), Spacer(1, 2)] + flow, bg=BLUE_BG, border=BLUE)


def recuerda(items):
    return box([P("<b>Lo que debes recordar</b>", "cellb"), Spacer(1, 2), bullets(items, "cell", GREEN)],
               bg=GREEN_BG, border=GREEN)


def code(lines):
    return box([P(l, "mono") for l in lines], bg=colors.white, width=CONTENT_W - 22)


def pasos(rows, widths=None):
    widths = widths or [11 * mm, 62 * mm, CONTENT_W - 73 * mm - 18]
    return table(rows, widths)


def apuntes():
    s = []
    s.append(band("Apuntes · Carpeta 2 · Modelos lineales generalizados",
                  "Para leer antes de los ejercicios · Clases 20 y 21 · lectura de ~60 min", C20))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("El tipo de Y manda", "Sí/no → logística. Conteo → Poisson. 3+ categorías → multinomial. Continua → lineal."),
                      ("Mismo molde", "Todo GLM = distribución + combinación lineal + enlace."),
                      ("Odds ≠ probabilidad", "odds = p / (1 − p). La logística es una recta en la escala log(odds)."),
                      ("e<super>β</super> es lo que se interpreta", "Logística: OR (multiplica las odds). Poisson: RR (multiplica la tasa)."),
                      ("Exactitud engaña", "Con clases desbalanceadas, reporta sensibilidad, especificidad y AUC."),
                      ("Offset y φ", "Offset para tasas; φ &gt; 1 indica sobredispersión → quasi-Poisson."),
                  ], C20)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios "
               "y una <b>pista</b> que apunta al notebook <font name='Mono'>C20-21_GLM.ipynb</font> o al cuaderno "
               "<font name='Mono'>03_Ejercicios_Cuaderno_GLM.pdf</font>.", "small"))

    # 1
    s.append(h("1", "Por qué la regresión lineal no alcanza"))
    s += para("En la regresión lineal, Y es continua y los errores son normales con varianza constante. Muchas preguntas "
              "reales no cumplen eso: <b>¿el cliente compra?</b> (sí/no), <b>¿cuántos reclamos llegan?</b> (conteo), "
              "<b>¿qué partido elige?</b> (categoría).")
    s += para("Si fuerzas una recta sobre una Y de 0 y 1, la recta no sabe que una probabilidad vive entre 0 y 1: termina "
              "prediciendo −0.2 o 1.3, que son valores imposibles. Además, la varianza de una proporción es p(1 − p), así que "
              "cambia con p y rompe el supuesto de varianza constante.")
    s.append(img("glm_porque.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(recuerda(["El problema no son las X: es que Y no es normal.",
                       "La solución no es abandonar la recta, sino ponerla en otra escala (la del enlace)."]))

    # 2
    s.append(h("2", "Los 3 componentes de un GLM"))
    s += para("Piensa en un GLM como un molde con tres piezas intercambiables:")
    s.append(table([["Pieza", "Qué es", "Quién la decide"],
                    ["1 · Distribución", "Cómo se comporta Y: Normal, Binomial, Poisson, Multinomial", "El TIPO de Y (no tú)"],
                    ["2 · Predictor lineal", "η = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + β<sub>2</sub>x<sub>2</sub> + …", "Tú eliges las X"],
                    ["3 · Enlace g", "Traduce la media de Y a la escala de η: g(μ) = η", "Tú, entre los compatibles con la distribución"]],
                   [34 * mm, 90 * mm, CONTENT_W - 124 * mm]))
    s.append(Spacer(1, 5))
    s += para("La profesora corrigió a un alumno justo en este punto: <b>la distribución depende de Y; el enlace se elige</b>. "
              "Para una binomial puedes usar logit, probit o cloglog. En clase, sus AIC fueron 1753.97, 1753.49 y 1754.25: "
              "casi iguales. Se suele preferir logit porque sus coeficientes se interpretan como odds ratios.")
    s.append(ejemplo("elegir distribución y enlace", [
        table([["Pregunta", "Tipo de Y", "Distribución", "Enlace"],
               ["¿Un correo es spam?", "binaria", "Binomial", "logit"],
               ["N° de accidentes por mes en una autopista", "conteo", "Poisson", "log"],
               ["Accidentes por cada 1000 autos que pasan", "conteo con exposición", "Poisson + offset", "log"],
               ["Tipo de plan contratado (básico, medio, premium)", "3 categorías con orden", "Multinomial ordinal", "logit acumulativo"],
               ["Temperatura máxima del día", "continua", "Normal", "identidad"]],
              [70 * mm, 36 * mm, 34 * mm, CONTENT_W - 140 * mm - 18])]))
    s.append(Spacer(1, 5))
    s.append(pista("el Bloque 1 del cuaderno", [P("Para cada situación, hazte dos preguntas en orden: <b>¿qué valores puede tomar Y?</b> "
                                                  "y <b>¿hay una «exposición» distinta entre observaciones</b> (población, tiempo, clientes)? "
                                                  "Si la hay, es Poisson con offset.", "cell")]))
    s.append(Spacer(1, 8))

    # 3
    s.append(h("3", "Probabilidad, odds y logit"))
    s += para("Son tres formas de decir lo mismo. Si la probabilidad de lluvia es 0.8, las <b>odds</b> son 0.8 / 0.2 = 4: "
              "«4 a 1 a favor de que llueva». El <b>logit</b> es el logaritmo de las odds: log(4) = 1.386.")
    s.append(img("glm_odds.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(KeepTogether([ejemplo("ida y vuelta con p = 0.6", [
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "odds = p / (1 − p) = 0.6 / 0.4", "1.5"],
               ["2", "logit = ln(odds) = ln(1.5)", "0.405"],
               ["3", "De vuelta: odds = e<super>logit</super> = e<super>0.405</super>", "1.5"],
               ["4", "p = odds / (1 + odds) = 1.5 / 2.5", "0.6"]]),
        Spacer(1, 3),
        P("Atajo útil: <b>p = 1 / (1 + e<super>−logit</super>)</b> pasa directo del logit a la probabilidad.", "cell")]),
        Spacer(1, 5),
        pista("el Bloque 2 del cuaderno", [P("Usa los 4 pasos de la tabla. Ten a mano las teclas <b>ln</b> y <b>e<super>x</super></b> "
                                             "de la calculadora. Para revisar tu resultado: si p &lt; 0.5, el logit debe ser negativo.", "cell")])]))

    # 4
    s.append(h("4", "Leer una regresión logística"))
    s += para("La logística ajusta <b>log(odds) = β<sub>0</sub> + β<sub>1</sub>x</b>. Como la recta vive en escala log-odds, los "
              "coeficientes se leen así:")
    s.append(bullets(["<b>β<sub>1</sub></b>: cuánto suben las log-odds por unidad de x. Es difícil de imaginar.",
                      "<b>e<super>β1</super> = OR</b>: por cuánto se <b>multiplican</b> las odds por unidad de x. Esto es lo que se reporta.",
                      "<b>(OR − 1) · 100</b>: el % de cambio en las odds.",
                      "<b>Por k unidades</b>: OR<sub>k</sub> = e<super>k·β1</super> (no k · OR)."], "body", C20))
    s.append(Spacer(1, 4))
    s.append(ejemplo("compras en una tienda online", [
        P("Modelo: log(odds de comprar) = −3 + 0.04 · minutos en el sitio.", "cell"),
        Spacer(1, 2),
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["OR por minuto", "e<super>0.04</super>", "1.041 → +4.1% en las odds"],
               ["OR por 10 minutos", "e<super>10 · 0.04</super> = e<super>0.4</super>", "1.492 → +49%"],
               ["P(comprar) con 50 min", "η = −3 + 0.04·50 = −1 → p = 1/(1 + e<super>1</super>)", "0.269"],
               ["P(comprar) con 0 min", "η = −3 → p = 1/(1 + e<super>3</super>)", "0.047"]],
              [36 * mm, 76 * mm, CONTENT_W - 112 * mm - 18]),
        Spacer(1, 3),
        P("Observa: por 10 minutos el OR es 1.492, no 10 · 1.041. Los efectos se <b>multiplican</b>, no se suman.", "cell")]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Odds ratio ≠ cambio en probabilidad</b>", "cellb"),
                  P("Con wcgs, pasar de 0 a 20 cigarrillos multiplica las odds por 1.59 (+59%). Pero la probabilidad pasa de 0.061 a "
                    "0.093: sube 3.2 puntos porcentuales. Ambas cosas son ciertas; solo hay que decir cuál estás reportando.", "cell")],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 5))
    s.append(pista("la Parte A del notebook y el Bloque 3 del cuaderno",
                   [P("Para el OR usa <font name='Mono'>np.exp(m.params)</font>. Para la probabilidad a mano, calcula primero η con los "
                      "coeficientes y después aplica 1 / (1 + e<super>−η</super>). Compara con <font name='Mono'>m.predict()</font>: deben coincidir.", "cell")]))
    s.append(Spacer(1, 8))

    # 5
    s.append(h("5", "Comparar modelos: deviance, prueba LR y AIC"))
    s += para("La <b>deviance</b> mide cuánto le falta a un modelo para ajustar perfecto: menos deviance, mejor ajuste. "
              "Cuando un modelo contiene al otro (modelos anidados), la <b>diferencia de deviances</b> sigue una χ² con tantos "
              "grados de libertad como parámetros agregaste. Esa es la prueba de razón de verosimilitud (LR).")
    s.append(ejemplo("¿agregar la edad mejora el modelo de wcgs?", [
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "Deviance de chd ~ cigs", "1750.0"],
               ["2", "Deviance de chd ~ cigs + age", "1705.9"],
               ["3", "LR = diferencia", "44.1"],
               ["4", "gl = parámetros agregados (solo age)", "1"],
               ["5", "Comparar con χ²<sub>crítico</sub>(1 gl, α = 0.05) = 3.84", "44.1 &gt; 3.84 → p ≈ 3·10<super>−11</super>"]]),
        Spacer(1, 2),
        P("Conclusión: la edad mejora significativamente el ajuste. El <b>AIC</b> sirve también para modelos <b>no</b> "
          "anidados: menor AIC, mejor equilibrio entre ajuste y simplicidad.", "cell")]))

    # 6
    s.append(h("6", "Evaluar un clasificador"))
    s += para("La logística entrega una probabilidad. Para decir «sí» o «no» se elige un <b>umbral</b>: si p ≥ umbral, se predice 1. "
              "Después se cruza lo predicho con lo real en una <b>matriz de confusión</b>.")
    s.append(ejemplo("una prueba de detección (200 personas)", [
        table([["", "Predicho: enfermo", "Predicho: sano"],
               ["Real: enfermo (50)", "VP = 40", "FN = 10"],
               ["Real: sano (150)", "FP = 30", "VN = 120"]],
              [44 * mm, 50 * mm, 50 * mm]),
        Spacer(1, 3),
        pasos([["Métrica", "Pregunta que responde", "Cálculo y resultado"],
               ["Exactitud", "¿Cuántos acierto en total?", "(40 + 120) / 200 = 0.80"],
               ["Sensibilidad", "De los enfermos, ¿a cuántos detecto?", "40 / 50 = 0.80"],
               ["Especificidad", "De los sanos, ¿a cuántos descarto?", "120 / 150 = 0.80"],
               ["Precisión", "De los que marco enfermos, ¿cuántos lo están?", "40 / 70 = 0.57"]],
              [28 * mm, 70 * mm, CONTENT_W - 98 * mm - 18])]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>La trampa de la exactitud (ejemplo de la clase)</b>", "cellb"),
                  P("Con wcgs y umbral 0.5, el modelo dijo «sano» a todos: VN = 881, FN = 66, VP = FP = 0. Exactitud = 881 / 947 = 93%, "
                    "pero sensibilidad = 0 / 66 = 0%. Solo el 8% está enfermo, así que decir siempre «no» ya acierta 92%. "
                    "Con el umbral de Youden (0.069) se detectó a 40 de 66 (61%), a costa de 397 falsas alarmas.", "cell")],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 5))
    s.append(img("glm_clasif.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook y el Bloque 4 del cuaderno",
                   [P("Ubica primero las 4 celdas (VP, FP, FN, VN) y escríbelas aparte. Todas las métricas son una celda dividida por una "
                      "suma de celdas: la fila «real» para sensibilidad y especificidad, la columna «predicho» para precisión.", "cell")]))

    # 7
    s.append(h("7", "Poisson: conteos, tasas y offset"))
    s += para("Para conteos se modela <b>log(μ) = η</b>. Al exponenciar, μ = e<super>η</super> siempre es positivo, como debe ser un conteo. "
              "Los coeficientes se leen como <b>razones de tasas</b>: e<super>β</super> dice por cuánto se multiplica la tasa.")
    s += para("El <b>offset</b> resuelve un problema práctico. Si una ciudad tiene el triple de habitantes, tendrá más casos aunque su "
              "riesgo sea igual. Agregando log(población) con coeficiente fijo 1, el modelo pasa a describir <b>casos por habitante</b>.")
    s.append(ejemplo("reclamos en sucursales", [
        P("Modelo: log(reclamos) = log(clientes) − 6 + 0.5 · (zona norte).", "cell"),
        Spacer(1, 2),
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Tasa en la zona sur (base)", "e<super>−6</super> por cliente", "0.00248 → 2.48 por 1000 clientes"],
               ["RR norte vs sur", "e<super>0.5</super>", "1.65 → tasa 65% mayor en el norte"],
               ["Reclamos esperados: sucursal norte, 2000 clientes", "2000 · e<super>−6 + 0.5</super>", "8.2 reclamos"]],
              [52 * mm, 50 * mm, CONTENT_W - 102 * mm - 18])]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Sobredispersión</b>", "cellb"),
                  P("La Poisson asume media = varianza. Se revisa con φ = X²<sub>Pearson</sub> / gl. Con φ ≈ 1 el modelo va bien. Con φ &gt; 1 "
                    "(en clase, 1.50) los errores estándar salen demasiado chicos y los p-valores demasiado optimistas. La quasi-Poisson "
                    "multiplica los errores estándar por √φ: en Python, <font name='Mono'>.fit(scale='X2')</font>.", "cell")],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 5))
    s.append(pista("la Parte C del notebook y el Bloque 5 del cuaderno",
                   [P("Para una tasa esperada, suma los coeficientes que aplican a esa fila (intercepto + ciudad + edad) y exponencia. "
                      "Para los casos esperados, multiplica esa tasa por la población. Las categorías de referencia (Fredericia, 40–54) "
                      "no suman nada: su coeficiente es 0.", "cell")]))
    s.append(Spacer(1, 8))

    # 8
    s.append(h("8", "Multinomial: más de dos categorías"))
    s += para("Con J categorías se elige una de <b>referencia</b> y se ajusta una ecuación logit para cada una de las otras J − 1. "
              "Cada ecuación compara «esta categoría vs la referencia». Para obtener probabilidades se usa la fórmula de abajo, "
              "con η = 0 para la referencia.")
    s.append(ejemplo("tres medios de transporte (A = referencia)", [
        P("Para cierta persona: η<sub>B</sub> = 0.5 y η<sub>C</sub> = −1.", "cell"),
        Spacer(1, 2),
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "Exponenciar: e<super>0</super>, e<super>0.5</super>, e<super>−1</super>", "1, 1.649, 0.368"],
               ["2", "Sumar", "3.017"],
               ["3", "Dividir cada uno por la suma", "A 0.331 · B 0.547 · C 0.122"],
               ["4", "Revisar: ¿suman 1?", "0.331 + 0.547 + 0.122 = 1"]])]))
    s.append(Spacer(1, 5))
    s.append(pista("la Parte D del notebook y el Bloque 6 del cuaderno",
                   [P("Calcula η para cada categoría que NO es la referencia con sus propios coeficientes. Luego aplica los 4 pasos. "
                      "Si tus probabilidades no suman 1, revisa que hayas incluido el 1 (= e<super>0</super>) de la referencia.", "cell")]))

    # 9 cheat sheet
    s.append(h("9", "Chuleta de código (statsmodels)"))
    s.append(table([
        ["Quiero…", "Código"],
        ["Importar", "import statsmodels.api as sm · import statsmodels.formula.api as smf"],
        ["Logística", "m = smf.logit('y ~ x1 + x2', d).fit()"],
        ["Logística (forma GLM)", "smf.glm('y ~ x', d, family=sm.families.Binomial()).fit()"],
        ["Odds ratios + IC", "np.exp(m.params) · np.exp(m.conf_int())"],
        ["Probabilidad predicha", "m.predict(pd.DataFrame({'x1': [..], 'x2': [..]}))"],
        ["Prueba LR (anidados)", "LR = m1.deviance − m2.deviance; stats.chi2.sf(LR, gl)"],
        ["Poisson con offset", "smf.glm('y ~ x', d, family=sm.families.Poisson(), offset=np.log(d.expo)).fit()"],
        ["φ (sobredispersión)", "m.pearson_chi2 / m.df_resid"],
        ["Quasi-Poisson", "....fit(scale='X2')"],
        ["Multinomial", "smf.mnlogit('y ~ x', d).fit()  (y = 0, 1, 2; 0 = referencia)"],
        ["Matriz de confusión", "confusion_matrix(y_real, (prob >= umbral).astype(int))"],
        ["AUC y ROC", "roc_auc_score(y_real, prob) · roc_curve(y_real, prob)"],
    ], [44 * mm, CONTENT_W - 44 * mm]))
    s.append(Spacer(1, 8))
    s.append(h("10", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Leer β de la logística como cambio en probabilidad", "β está en escala log-odds", "Reportar e<super>β</super> (OR) o probabilidades predichas"],
        ["OR por 10 unidades = 10 · OR", "Los efectos se multiplican", "e<super>10·β</super>"],
        ["Confiar en la exactitud", "Con clases desbalanceadas engaña", "Sensibilidad, especificidad, AUC"],
        ["Poisson sin offset con exposiciones distintas", "Compara conteos, no riesgos", "offset = log(exposición)"],
        ["Ignorar φ &gt; 1", "p-valores demasiado optimistas", "Quasi-Poisson o binomial negativa"],
        ["Evaluar con los datos de entrenamiento", "Sobreestima el rendimiento", "Separar train / test"],
    ], [52 * mm, 50 * mm, CONTENT_W - 102 * mm], header_color=RED))
    s.append(Spacer(1, 8))
    s.append(recuerda(["Orden sugerido: lee estos apuntes → haz los Bloques 1–3 del cuaderno → Partes A y B del notebook → "
                       "Bloques 4–6 → Partes C y D → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/03_Apuntes_GLM_C20-21.pdf", s, "Apuntes · Carpeta 2 · GLM · Clases 20–21")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0891B2">Bloque {n}</font>  {title}  <font size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow.append(P(intro, "cell"))
        flow.append(Spacer(1, 3))
    flow.append(numbered([("", t) for t in items], C20))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Carpeta 2 · GLM",
                  "Para resolver en papel, con calculadora (teclas ln y eˣ) · soluciones paso a paso al final", C20))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe en el cuaderno el <b>número del bloque y del ejercicio</b>, y luego tu desarrollo completo, no solo el resultado.",
                           "Redondea a 3 o 4 decimales. Pequeñas diferencias de redondeo con las soluciones son normales.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> requiere 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos.",
                           "Corrige con las soluciones de las últimas páginas y marca con color lo que tuviste mal. "
                           "Esos ejercicios se repiten en el repaso del domingo."], "cell", C20)],
                 bg=SOFT, border=C20))
    s.append(Spacer(1, 4))
    s += bloque(1, "Elegir el modelo", "<font name='Sym'>★</font>", "Para cada situación indica: tipo de Y, distribución, enlace y si necesita offset.", [
        "Un banco quiere predecir si un cliente pagará su crédito (sí/no).",
        "Un hospital cuenta las infecciones por mes en cada piso; los pisos tienen distinta cantidad de camas.",
        "Una ciudad registra el medio de transporte que usa cada persona: auto, bus o bicicleta.",
        "Una encuesta mide la satisfacción como mala, regular o buena.",
        "Se registra el peso de recién nacidos según las semanas de gestación.",
        "Se cuentan los goles por partido de un equipo.",
    ])
    s += bloque(2, "Probabilidad, odds y logit", "<font name='Sym'>★</font>", None, [
        "Convierte p = 0.2, p = 0.75 y p = 0.05 a odds y a logit.",
        "Si las odds son 3, ¿cuál es la probabilidad?",
        "Si el logit es −1, ¿cuáles son las odds y la probabilidad?",
        "Sin calcular: si p &lt; 0.5, ¿el logit es positivo o negativo? ¿Por qué?",
    ])
    s += bloque(3, "Interpretar una logística (datos reales de wcgs)", "<font name='Sym'>★★</font>",
                "Modelo de clase: log(odds de chd) = −2.742 + 0.0232 · cigs.", [
        "Calcula las odds y la probabilidad de chd para alguien que no fuma.",
        "Calcula la probabilidad de chd para alguien que fuma 20 cigarrillos diarios.",
        "Calcula el OR por 1 cigarrillo y por 10 cigarrillos. Interpreta cada uno en una frase con %.",
        "El material de clase dice que un OR de 1.2614 significa «35% mayores». ¿Qué está mal? ¿Cuál es la frase correcta?",
        "Con edad: log(odds) = −6.360 + 0.0239 · cigs + 0.0764 · age. Calcula la probabilidad para un hombre de 50 años que fuma 20 "
        "cigarrillos, y el OR por 10 años más de edad.",
        "<font name='Sym'>★★★</font> Al agregar la edad, la deviance baja de 1750.0 a 1705.9. Plantea H0, calcula el estadístico LR, indica los grados de "
        "libertad y concluye (χ² crítico con 1 gl y α = 0.05: 3.84).",
    ])
    s += bloque(4, "Matriz de confusión", "<font name='Sym'>★★</font>", None, [
        "Matriz de clase con umbral 0.5: VN = 881, FP = 0, FN = 66, VP = 0. Calcula exactitud, sensibilidad y especificidad. "
        "¿Por qué la exactitud es engañosa aquí?",
        "Con el umbral de Youden: VN = 484, FP = 397, FN = 26, VP = 40. Calcula las mismas tres métricas y la precisión. "
        "¿Qué ganaste y qué perdiste?",
        "Una prueba tiene VP = 90, FN = 10, FP = 45, VN = 855. Calcula las 4 métricas.",
        "<font name='Sym'>★★★</font> Para detectar una enfermedad grave pero tratable, ¿preferirías alta sensibilidad o alta especificidad? Justifica en "
        "dos líneas y di si subirías o bajarías el umbral.",
    ])
    s += bloque(5, "Poisson con offset (cáncer de pulmón)", "<font name='Sym'>★★</font>",
                "Coeficientes de clase: intercepto −5.632 · Kolding −0.3715 · Horsens −0.3301 · edad 70–74 1.857 · edad 55–59 1.101. "
                "Referencias: Fredericia y 40–54.", [
        "Calcula la tasa esperada por 1000 habitantes para Fredericia, 40–54 años.",
        "Calcula la tasa por 1000 para Kolding, 70–74 años, y los casos esperados si esa población es de 535 personas.",
        "Calcula e interpreta el RR de Kolding y el de la edad 55–59.",
        "¿Por qué el modelo incluye log(población) como offset? Responde con un ejemplo de dos ciudades.",
        "<font name='Sym'>★★★</font> X² de Pearson = 22.56 con 15 gl. Calcula φ, interprétalo y di qué modelo usarías y qué cambiaría en los resultados.",
    ])
    s += bloque(6, "Multinomial (nes96)", "<font name='Sym'>★★★</font>",
                "Referencia: Demócrata. η<sub>Rep</sub> = −0.863 + 0.0068 · edad + 0.1095 · educ. "
                "η<sub>Ind</sub> = −2.385 + 0.0019 · edad − 0.0845 · educ. (educ va de 0 = MS a 6 = MAdeg).", [
        "Calcula η<sub>Rep</sub> y η<sub>Ind</sub> para una persona de 45 años con educ = 5 (BAdeg).",
        "Calcula las tres probabilidades y verifica que suman 1.",
        "Calcula e<super>0.1095</super> e interprétalo en una frase.",
        "El modelo nunca predice «Independiente» como clase más probable. Explica por qué en dos líneas.",
    ])
    s += bloque(7, "Desafío teórico", "<font name='Sym'>★★★</font>", None, [
        "Partiendo de log(odds(x)) = β<sub>0</sub> + β<sub>1</sub>x, demuestra que odds(x + 1) / odds(x) = e<super>β1</super>.",
        "Un coeficiente vale −0.5 en una logística. Calcula el OR, di si es factor de riesgo o protector, y da el % de cambio.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------- soluciones
    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows, widths=None):
        widths = widths or [14 * mm, CONTENT_W - 14 * mm]
        return [P(title, "h3"), table([["#", "Solución"]] + rows, widths, header_color=GREEN_C), Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "Binaria → Binomial, enlace logit (regresión logística). Sin offset."],
        ["2", "Conteo con exposición distinta → Poisson, enlace log, offset = log(n° de camas)."],
        ["3", "3 categorías sin orden → Multinomial nominal (logit generalizado)."],
        ["4", "3 categorías con orden → Multinomial ordinal (logit acumulativo)."],
        ["5", "Continua → Normal, enlace identidad (regresión lineal)."],
        ["6", "Conteo, misma exposición (1 partido) → Poisson, enlace log, sin offset."]])
    s += sol("Bloque 2", [
        ["1", "p = 0.2 → odds 0.25, logit −1.386 · p = 0.75 → odds 3, logit 1.099 · p = 0.05 → odds 0.0526, logit −2.944."],
        ["2", "p = odds / (1 + odds) = 3 / 4 = 0.75."],
        ["3", "odds = e<super>−1</super> = 0.368 → p = 0.368 / 1.368 = 0.269."],
        ["4", "Negativo: p &lt; 0.5 → odds &lt; 1 → ln(odds) &lt; 0."]])
    s += sol("Bloque 3", [
        ["1", "odds = e<super>−2.742</super> = 0.0644 → p = 0.0644 / 1.0644 = 0.0605 (6%)."],
        ["2", "η = −2.742 + 0.0232 · 20 = −2.278 → p = 1 / (1 + e<super>2.278</super>) = 0.093 (9.3%)."],
        ["3", "OR<sub>1</sub> = e<super>0.0232</super> = 1.0235: cada cigarrillo extra aumenta las odds 2.35%. "
              "OR<sub>10</sub> = e<super>0.232</super> = 1.2614: 10 cigarrillos más aumentan las odds 26%."],
        ["4", "1.2614 significa odds 1.26 veces, o sea +26%. «1.35 veces / 35%» no corresponde a ese OR. Frase correcta: «quien fuma "
              "10 cigarrillos diarios más tiene odds de chd 1.26 veces las del otro, es decir, 26% mayores»."],
        ["5", "η = −6.360 + 0.0239 · 20 + 0.0764 · 50 = −2.06 → p = 1 / (1 + e<super>2.06</super>) = 0.112 (11%). "
              "OR<sub>10 años</sub> = e<super>0.764</super> = 2.15: 10 años más duplican (×2.15) las odds."],
        ["6", "H0: el coeficiente de la edad es 0. LR = 1750.0 − 1705.9 = 44.1 con 1 gl. 44.1 &gt; 3.84 → se rechaza H0 "
              "(p ≈ 3·10<super>−11</super>): la edad mejora el modelo."]])
    s.append(PageBreak())
    s += sol("Bloque 4", [
        ["1", "Exactitud = 881 / 947 = 0.930 · Sensibilidad = 0 / 66 = 0 · Especificidad = 881 / 881 = 1. Engaña porque el 93% "
              "de los casos son sanos: decir «sano» a todos ya da 93% sin detectar a nadie."],
        ["2", "Exactitud = (40 + 484) / 947 = 0.553 · Sensibilidad = 40 / 66 = 0.606 · Especificidad = 484 / 881 = 0.549 · "
              "Precisión = 40 / 437 = 0.092. Ganaste detectar al 61% de los enfermos; perdiste exactitud y hay muchas falsas alarmas."],
        ["3", "Exactitud = 945 / 1000 = 0.945 · Sensibilidad = 90 / 100 = 0.90 · Especificidad = 855 / 900 = 0.95 · "
              "Precisión = 90 / 135 = 0.667."],
        ["4", "Alta sensibilidad: un falso negativo (enfermo no detectado) es más grave que una falsa alarma que se descarta con "
              "otro examen. Bajaría el umbral."]])
    s += sol("Bloque 5", [
        ["1", "tasa = e<super>−5.632</super> = 0.00358 por persona → 3.58 por 1000."],
        ["2", "η = −5.632 − 0.3715 + 1.857 = −4.147 → tasa = e<super>−4.147</super> = 0.0158 → 15.8 por 1000. "
              "Casos esperados = 535 · 0.0158 = 8.5."],
        ["3", "RR Kolding = e<super>−0.3715</super> = 0.69: su tasa es 31% menor que la de Fredericia, a igual edad. "
              "RR 55–59 = e<super>1.101</super> = 3.01: tasa 3 veces la de 40–54, en la misma ciudad."],
        ["4", "Ciudad A con 10.000 habitantes y 20 casos vs ciudad B con 1.000 y 5 casos: A tiene más casos, pero su tasa (2 por 1000) "
              "es menor que la de B (5 por 1000). El offset hace que el modelo compare tasas."],
        ["5", "φ = 22.56 / 15 = 1.50 &gt; 1 → sobredispersión leve. Usaría quasi-Poisson: mismos coeficientes, errores estándar × √1.5 "
              "≈ 1.23, p-valores más grandes (en clase, Kolding deja de ser significativa)."]])
    s += sol("Bloque 6", [
        ["1", "η<sub>Rep</sub> = −0.863 + 0.306 + 0.548 = −0.010 · η<sub>Ind</sub> = −2.385 + 0.086 − 0.423 = −2.722."],
        ["2", "e<super>0</super> = 1, e<super>−0.010</super> = 0.990, e<super>−2.722</super> = 0.066; suma = 2.056 → "
              "Dem 0.486 · Rep 0.482 · Ind 0.032. Suman 1."],
        ["3", "e<super>0.1095</super> = 1.116: cada nivel educacional adicional multiplica P(Rep) / P(Dem) por 1.116 (+11.6%)."],
        ["4", "Independiente es solo el 4% de los datos y ninguna X la separa bien: su probabilidad casi nunca supera a las de "
              "Demócrata y Republicano, así que nunca es la más alta."]])
    s += sol("Bloque 7", [
        ["1", "odds(x) = e<super>β0 + β1x</super>, odds(x + 1) = e<super>β0 + β1x + β1</super>. Dividiendo: "
              "e<super>β0 + β1x + β1</super> / e<super>β0 + β1x</super> = e<super>β1</super>. No depende de x: el OR es constante."],
        ["2", "OR = e<super>−0.5</super> = 0.607 &lt; 1 → factor protector: las odds bajan 39.3% por unidad de x."]])
    build("apuntes/03_Ejercicios_Cuaderno_GLM.pdf", s, "Ejercicios para el cuaderno · Carpeta 2 · GLM")


if __name__ == "__main__":
    guia()
    apuntes()
    cuaderno()
    print("OK")
