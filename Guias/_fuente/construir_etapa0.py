"""Construye la guía visual de la Etapa 0 (nivelación estadística)."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, P, RED,
                           RED_BG, SOFT, band, box, build, bullets, img, numbered, table,
                           two_cols)

E0 = colors.HexColor("#475569")
SYM = '<font name="Sym">☐</font>'


def session_page(s, num, fecha, titulo, idea, conceptos, figura, pie, formulas, py,
                 interpretar, cuidado, ejercicio):
    s.append(band(f"Sesión {num} · {titulo}", f"{fecha} · ~95 min · dataset: ENS.csv", E0))
    s.append(Spacer(1, 7))
    s.append(box([P(idea, "big")], bg=SOFT, border=E0))
    s.append(Spacer(1, 6))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, E0))
    s.append(Spacer(1, 4))
    s.append(img(figura, width=CONTENT_W * 0.97))
    s.append(P(pie, "small"))
    s.append(Spacer(1, 5))
    half = (CONTENT_W - 6 * mm) * 0.5
    left = [P("Fórmulas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=half),
            Spacer(1, 5),
            P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=half)]
    right = [P("Con los datos reales", "h3"), bullets(interpretar, "cell", E0),
             Spacer(1, 5),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado],
                 bg=AMBER_BG, border=AMBER, width=half),
             Spacer(1, 5),
             box([P("<b>Ejercicio de cierre</b> (solución al final)", "cellb"), P(ejercicio, "cell")],
                 bg=GREEN_BG, border=GREEN, width=half)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


def etapa0_pdf():
    s = []
    # ------------------------------------------------------------ overview
    s.append(band("Etapa 0 · Nivelación estadística",
                  "7 sesiones · 6 al 18 de octubre · la base de las Clases 17 a 24", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("Por qué esta etapa existe", "cellb"), Spacer(1, 3),
                  P("ANOVA, regresión, GLM, A/B testing y estadística bayesiana usan el mismo lenguaje: "
                    "<b>muestras, error estándar, intervalos y p-valores</b>. Sin ese lenguaje, las clases se pueden "
                    "<i>seguir</i> pero no <i>entender</i>. Son 2 semanas que ahorran muchas más después.", "body")],
                 bg=colors.HexColor("#F1F5F9"), border=E0))
    s.append(Spacer(1, 6))
    s.append(P("El camino", "h2"))
    s.append(img("e0_mapa.png"))
    s.append(Spacer(1, 4))
    s.append(P("Calendario", "h2"))
    s.append(table([
        ["Sesión", "Día", "Tema", "Pregunta que responde"],
        ["0.1", "mar 6 oct", "Describir los datos", "¿Cómo son mis datos?"],
        ["0.2", "mié 7 oct", "Distribuciones y la normal", "¿Qué valores son comunes y cuáles raros?"],
        ["0.3", "vie 9 oct", "Muestras y error estándar", "¿Cuánto cambiaría mi promedio con otra muestra?"],
        ["Repaso", "dom 11 oct", "Sesiones 0.1 a 0.3", "Mini evaluación de 6 preguntas"],
        ["0.4", "lun 12 oct", "Intervalos de confianza", "¿Entre qué valores está la media real?"],
        ["0.5", "mar 13 oct", "Prueba de hipótesis y p-valor", "¿La diferencia es real o es ruido?"],
        ["0.6", "mié 14 oct", "Errores, potencia y tamaño de efecto", "¿Cuánto me puedo equivocar?"],
        ["0.7", "vie 16 oct", "Pruebas t, proporción y χ²", "¿Qué prueba uso en cada caso?"],
        ["Repaso", "dom 18 oct", "Toda la etapa", "Evaluación final + criterio de dominio"],
    ], [16 * mm, 22 * mm, 58 * mm, CONTENT_W - 96 * mm]))
    s.append(Spacer(1, 6))
    left = [P("Un solo dataset para toda la etapa", "h3"),
            P("<b>ENS.csv</b> (carpeta de las Clases 17–19): 350 personas con "
              "<b>COLESTEROL</b> (mg/dL), <b>EDAD</b> (Joven, Adulto, Adulto Mayor; 4 vacíos), "
              "<b>FUMA</b> y <b>DEPORTE</b> (Sí/No). Así te concentras en los conceptos y no en "
              "entender datos nuevos cada día.", "cell")]
    right = [P("Cambio respecto al roadmap", "h3"),
             P("<b>Correlación</b> sale de esta etapa: la Clase 18 la enseña completa. "
               "Eso libera una sesión y la etapa queda en 7.", "cell"),
             Spacer(1, 3),
             P("<b>Para las sesiones 0.6 en adelante</b> necesitas instalar statsmodels: "
               "<font name='Mono'>pip install statsmodels</font>", "cell")]
    s.append(two_cols(left, right))
    s.append(PageBreak())

    # ------------------------------------------------------------ 0.1
    session_page(
        s, "0.1", "Martes 6 de octubre", "Describir los datos",
        "Antes de cualquier modelo hay que conocer los datos: qué tipo es cada variable, "
        "dónde está su centro y cuánto se dispersa.",
        [("Tipos de variables", "Numérica (continua o discreta) o categórica (nominal u ordinal). El tipo decide qué análisis puedes hacer. EDAD en ENS es categórica ordinal."),
         ("Centro", "Media y mediana. La media se mueve con los valores extremos; la mediana no. Si difieren mucho, hay asimetría u outliers."),
         ("Dispersión", "Varianza, desviación estándar (en las mismas unidades que los datos) y rango intercuartílico (IQR = Q3 − Q1)."),
         ("Boxplot", "Resume 5 números: mínimo, Q1, mediana, Q3 y máximo. Un outlier está a más de 1.5·IQR fuera de la caja."),
         ("Datos faltantes", "EDAD tiene 4 vacíos (346 de 350). Cuéntalos siempre antes de analizar.")],
        "e0_s01.png",
        "Media y mediana casi iguales → distribución casi simétrica. Los jóvenes tienen la caja claramente más abajo. "
        "¿Es una diferencia real o azar? Eso lo responde la sesión 0.7.",
        ["x̄ = Σx / n",
         "s² = Σ(x − x̄)² / (n − 1)",
         "s = √s²",
         "IQR = Q3 − Q1",
         "outlier: x < Q1 − 1.5·IQR",
         "      o x > Q3 + 1.5·IQR"],
        ["d.describe(include='all')",
         "d.isna().sum()",
         "d.groupby('EDAD').COLESTEROL",
         "  .agg(['mean','median','std'])",
         "d.FUMA.value_counts(normalize=True)"],
        ["Media 192.8 · mediana 189.8 · s = 43.3",
         "Joven 163 · Adulto 201 · Adulto Mayor 208",
         "34% fuma · 19% hace deporte"],
        ["pandas calcula la desviación estándar con n − 1 (43.28); numpy usa n por defecto (43.22). "
         "Con muestras chicas la diferencia importa: usa ddof=1 en numpy."],
        "Con salarios.xlsx, calcula media, mediana y desviación estándar del salario por industria. "
        "¿Qué industria tiene salarios más dispersos?")

    # ------------------------------------------------------------ 0.2
    session_page(
        s, "0.2", "Miércoles 7 de octubre", "Distribuciones y la normal",
        "Una distribución dice qué valores son comunes y cuáles son raros. La normal es la más "
        "importante porque aparece una y otra vez, sobre todo al promediar.",
        [("Distribución de probabilidad", "El área bajo la curva es la probabilidad. El área total es 1."),
         ("Normal", "Simétrica y definida solo por μ (centro) y σ (ancho). Regla 68–95–99.7."),
         ("Puntaje z", "Cuántas desviaciones estándar está un valor de la media. Permite comparar variables con distintas unidades."),
         ("Verificar normalidad", "Histograma + QQ-plot. Shapiro-Wilk en ENS da p = 0.007 aunque el histograma se ve casi normal: con n grande, el test detecta desvíos mínimos. Mira el gráfico, no solo el p-valor."),
         ("Otras que verás", "t de Student (como la normal, con colas más gruesas; para muestras chicas), chi-cuadrado y binomial (conteo de éxitos).")],
        "e0_s02.png",
        "Izquierda: la regla 68–95–99.7. Derecha: el colesterol real con la normal que mejor se le ajusta. "
        "Se parecen bastante, pero no son idénticas.",
        ["z = (x − μ) / σ",
         "P(X > x) = 1 − Φ(z)",
         "Φ = área acumulada de la normal",
         "z = ±1.96 deja el 95% al centro"],
        ["(c - c.mean()) / c.std()",
         "stats.norm.cdf(1.32)   # 0.907",
         "stats.norm.ppf(0.975)  # 1.96",
         "stats.probplot(c, plot=plt)",
         "stats.shapiro(c)"],
        ["250 mg/dL → z = 1.32",
         "La normal predice 9.3% sobre 250",
         "En los datos reales es 9.7% → buen modelo"],
        ["La normal es un MODELO, no una ley de la naturaleza. Antes de usarla, verifica que se parezca a tus datos."],
        "Si el colesterol fuera Normal(193, 43), ¿qué porcentaje de personas tendría menos de 150 mg/dL? "
        "Compáralo con el porcentaje real de ENS.")

    # ------------------------------------------------------------ 0.3
    session_page(
        s, "0.3", "Viernes 9 de octubre", "Muestras y error estándar",
        "Nunca vemos la población completa, solo una muestra. El error estándar mide cuánto cambiaría "
        "nuestro promedio si tomáramos otra muestra.",
        [("Población vs muestra", "Parámetros (μ, σ) describen la población y son desconocidos. Estadísticos (x̄, s) se calculan en la muestra y los estiman."),
         ("Distribución muestral", "Si repitieras el muestreo muchas veces, las medias formarían su propia distribución."),
         ("Teorema Central del Límite", "Con n suficientemente grande (regla práctica: 30 o más), la media se distribuye aproximadamente normal, AUNQUE la población no lo sea."),
         ("Error estándar", "EE = s / √n: la desviación estándar de la media. Para reducirlo a la mitad hay que cuadruplicar n."),
         ("s vs EE", "s describe cuánto varían las PERSONAS. EE describe cuán preciso es el PROMEDIO.")],
        "e0_s03.png",
        "Simulación con una población muy asimétrica. Con n = 5 las medias todavía son asimétricas; con n = 30 ya "
        "son prácticamente normales y mucho más concentradas.",
        ["EE = σ / √n  ≈  s / √n",
         "x̄ ≈ Normal(μ, σ²/n)  si n es grande"],
        ["stats.sem(c)   # 2.31",
         "rng = np.random.default_rng(0)",
         "rng.choice(pob, (5000, n))",
         "   .mean(axis=1)"],
        ["s = 43.3 → las personas varían ±43 mg/dL",
         "EE = 43.3 / √350 = 2.31",
         "El PROMEDIO se conoce con precisión de ±2"],
        ["El TCL habla de la distribución de la MEDIA, no de los datos: no convierte tus datos en normales."],
        "¿Cuánto sería el error estándar del colesterol con n = 35? ¿Y con n = 3.500? ¿Cuántas veces más datos "
        "necesitas para tener 10 veces más precisión?")

    # ------------------------------------------------------------ 0.4
    session_page(
        s, "0.4", "Lunes 12 de octubre", "Intervalos de confianza",
        "Un intervalo de confianza entrega un rango de valores plausibles para el parámetro, en vez "
        "de un solo número que casi seguro no es exacto.",
        [("Puntual vs intervalo", "x̄ = 192.8 es una estimación puntual. El intervalo agrega cuánta incertidumbre tiene."),
         ("IC 95% de la media", "x̄ ± t · EE. Con n grande, t ≈ 1.96 (de la sesión 0.2)."),
         ("Interpretación correcta", "Si repitiéramos el muestreo muchas veces, el 95% de los intervalos construidos así contendrían μ. Eso es lo que muestra la figura."),
         ("Ancho", "Más n → más angosto. Más confianza (99%) → más ancho."),
         ("Puente con las pruebas", "Si el IC 95% no contiene un valor, una prueba bilateral con α = 0.05 rechazaría ese valor.")],
        "e0_s04.png",
        "Simulación: 40 muestras de 30 personas desde una población con la media y desviación del colesterol real. "
        "Cada línea es un IC 95%.",
        ["IC = x̄ ± t(n−1; 0.975) · s/√n",
         "t ≈ 1.96 si n es grande"],
        ["stats.t.interval(0.95, n - 1,",
         "    loc=c.mean(),",
         "    scale=stats.sem(c))"],
        ["IC 95% del colesterol: [188.2; 197.3]",
         "No contiene 200 → la sesión 0.7 lo confirma con una prueba t (p = 0.002)"],
        ["NO digas «hay 95% de probabilidad de que μ esté en este intervalo». μ es fija: lo que varía de muestra en "
         "muestra es el intervalo. Es sutil, pero lo preguntan."],
        "Calcula el IC 95% del colesterol para quienes hacen deporte y para quienes no. ¿Se solapan los intervalos? "
        "¿Qué sugiere eso?")

    # ------------------------------------------------------------ 0.5
    session_page(
        s, "0.5", "Martes 13 de octubre", "Prueba de hipótesis y p-valor",
        "Una prueba de hipótesis funciona como un juicio: H0 es inocente hasta que los datos aporten "
        "evidencia suficiente en su contra.",
        [("H0 y H1", "H0: no hay efecto o diferencia. H1: sí lo hay. Siempre se plantean ANTES de mirar el resultado."),
         ("Estadístico de prueba", "Mide cuán lejos están los datos de lo que predice H0, en unidades de error estándar: t = diferencia / EE."),
         ("p-valor", "Probabilidad de obtener un resultado igual o más extremo que el observado, SI H0 fuera cierta. Es el área naranja de la figura."),
         ("Decisión", "Si p < α (0.05), se rechaza H0. Si no, «no se rechaza» — NUNCA «se acepta H0»: ausencia de evidencia no es evidencia de ausencia."),
         ("Bilateral o unilateral", "Bilateral: «¿es distinto?» (lo habitual). Unilateral: «¿es mayor?», solo si la dirección se justifica antes de ver los datos.")],
        "e0_s05.png",
        "Fumadores vs no fumadores en ENS. Los fumadores tienen 5.3 mg/dL más en promedio, pero esa diferencia "
        "cabe perfectamente dentro de lo que produce el azar.",
        ["t = (x̄₁ − x̄₂) / EE_diferencia",
         "p < α  →  rechazar H0",
         "p ≥ α  →  no rechazar H0"],
        ["a = c[d.FUMA == 'Si']",
         "b = c[d.FUMA == 'No']",
         "stats.ttest_ind(a, b,",
         "                equal_var=False)"],
        ["Fuma: 196.3 · No fuma: 190.9",
         "t = 1.04 · p = 0.30",
         "→ no hay evidencia de que fumar cambie el colesterol en estos datos"],
        ["El p-valor NO es la probabilidad de que H0 sea cierta. Recuerda la alerta de la Clase 17: "
         "la profesora lo confunde con α."],
        "Plantea H0 y H1 para la pregunta «¿hacer deporte se asocia a un colesterol distinto?». "
        "¿Es una prueba bilateral o unilateral?")

    # ------------------------------------------------------------ 0.6
    session_page(
        s, "0.6", "Miércoles 14 de octubre", "Errores, potencia y tamaño de efecto",
        "Toda decisión con datos puede equivocarse de dos formas. Diseñar bien un estudio es "
        "controlar las dos.",
        [("Error tipo I (α)", "Rechazar H0 cuando es cierta: una falsa alarma. Lo fijas tú, normalmente en 0.05."),
         ("Error tipo II (β)", "No rechazar H0 cuando es falsa: un efecto real que no detectas."),
         ("Potencia = 1 − β", "Probabilidad de detectar un efecto que existe. Sube con más n, con efectos más grandes y con α mayor. Lo habitual es buscar 80%."),
         ("Significativo ≠ importante", "Con n enorme, diferencias mínimas salen significativas. Reporta siempre el tamaño del efecto, por ejemplo la d de Cohen."),
         ("Comparaciones múltiples", "Con 20 pruebas a α = 0.05, se espera ~1 falso positivo solo por azar. Por eso ANOVA usa Tukey.")],
        "e0_s06.png",
        "Si H0 es cierta (curva negra), el área roja son las falsas alarmas. Si H1 es cierta (curva azul), el área "
        "naranja son los efectos que se escapan. Mover el umbral cambia un error por el otro.",
        ["H0 cierta + rechazo   → Error I (α)",
         "H0 falsa + no rechazo → Error II (β)",
         "Potencia = 1 − β",
         "d = (x̄₁ − x̄₂) / s_combinada",
         "0.2 pequeño · 0.5 medio · 0.8 grande"],
        ["from statsmodels.stats.power \\",
         "    import TTestIndPower",
         "TTestIndPower().solve_power(",
         "    effect_size=0.43,",
         "    alpha=0.05, power=0.8)"],
        ["Deporte: d = 0.43 (efecto entre pequeño y medio)",
         "Para detectarlo con 80% de potencia se necesitan ~85 personas por grupo",
         "ENS tiene 68 que hacen deporte: la detectó igual (p = 0.0015)"],
        ["p = 0.049 y p = 0.051 son prácticamente la misma evidencia. 0.05 es una convención, no una frontera mágica."],
        "Haces 10 pruebas independientes con α = 0.05 y todas las H0 son ciertas. ¿Cuál es la probabilidad "
        "de obtener al menos un falso positivo?")

    # ------------------------------------------------------------ 0.7
    session_page(
        s, "0.7", "Viernes 16 de octubre", "Pruebas t, de proporción y χ²",
        "Elegir la prueba depende de dos preguntas: qué tipo de variable mides y cuántos grupos comparas.",
        [("t de una muestra", "¿La media es distinta de un valor? Colesterol vs 200: t = −3.13, p = 0.002 → sí, es menor."),
         ("t de dos muestras (Welch)", "¿Dos grupos difieren? Deporte vs no: −18.5 mg/dL, p = 0.0015 → sí. Usa Welch (equal_var=False): no asume varianzas iguales."),
         ("Prueba de proporción", "¿El % de fumadores (34.3%) es distinto de 30%? p = 0.09 → no hay evidencia."),
         ("Chi-cuadrado (χ²)", "¿Dos categóricas están asociadas? FUMA × DEPORTE: χ² = 4.18, p = 0.041 → sí. En esta muestra los fumadores hacen MÁS deporte (26% vs 16%). Sorprendente: así se ve una asociación que no implica causa."),
         ("Supuestos", "Observaciones independientes. t: normalidad aproximada o n grande (TCL). χ²: frecuencias esperadas de 5 o más en cada celda.")],
        "e0_s07.png",
        "Mapa de decisión. Para 3 o más grupos no se encadenan pruebas t (se inflaría el error tipo I): se usa ANOVA, "
        "el primer tema de la Clase 17.",
        ["t = (x̄ − μ₀) / (s/√n)",
         "χ² = Σ (observado − esperado)² / esperado"],
        ["stats.ttest_1samp(c, 200)",
         "stats.ttest_ind(a, b, equal_var=False)",
         "stats.binomtest(120, 350, p=0.30)",
         "tabla = pd.crosstab(d.FUMA, d.DEPORTE)",
         "stats.chi2_contingency(tabla)"],
        ["Antes de elegir: ¿la respuesta es numérica o categórica?",
         "Después: ¿cuántos grupos comparo?",
         "Al final: ¿se cumplen los supuestos?"],
        ["Reporta siempre tres cosas: el estadístico, el p-valor y la diferencia en unidades reales "
         "(ej. «18.5 mg/dL menos»). Solo el p-valor no dice cuán grande es el efecto."],
        "¿El colesterol difiere entre «Adulto» y «Adulto Mayor»? Elige la prueba, ejecútala y escribe la "
        "conclusión en una frase.")

    # ------------------------------------------------------------ cierre
    s.append(band("Cierre de la Etapa 0", "Errores clásicos, evaluación, soluciones y recursos", NAVY))
    s.append(Spacer(1, 7))
    s.append(P("Errores clásicos de interpretación", "h2"))
    s.append(table([
        ["Lo que se dice mal", "Lo correcto"],
        ["«p = 0.03: hay 3% de probabilidad de que H0 sea cierta»",
         "Si H0 fuera cierta, un resultado así o más extremo aparecería el 3% de las veces"],
        ["«p = 0.30: entonces H0 es verdadera»", "No hay evidencia suficiente para rechazarla. Eso no la prueba"],
        ["«Hay 95% de probabilidad de que μ esté en [188; 197]»",
         "El 95% de los intervalos construidos así contienen μ"],
        ["«Es significativo, entonces es importante»", "Mira el tamaño del efecto en unidades reales"],
        ["«El TCL hace que mis datos sean normales»", "El TCL habla de la distribución de la media, no de los datos"],
        ["«Están asociados, entonces uno causa el otro»", "Asociación no es causalidad (Clase 22: causalidad aplicada)"],
    ], [CONTENT_W * 0.45, CONTENT_W * 0.55], header_color=RED))
    s.append(Spacer(1, 7))
    s.append(P("Soluciones de los ejercicios de cierre", "h2"))
    s.append(table([
        ["Sesión", "Solución"],
        ["0.1", "Retail: media 2.90, mediana 2.80, s = 0.86 · Banca: 3.36 / 3.35 / 0.63 · Minería: 4.48 / 4.50 / 0.52. "
                "Retail es la más dispersa."],
        ["0.2", "z = (150 − 193) / 43 = −0.99 → la normal predice 16.2%. En los datos reales es 18.3%: "
                "la normal subestima un poco la cola baja."],
        ["0.3", "n = 35 → EE = 7.32 · n = 3.500 → EE = 0.73. Para 10 veces más precisión se necesitan 10² = 100 veces más datos."],
        ["0.4", "Deporte: [167.8; 188.0] · Sin deporte: [191.3; 201.4]. No se solapan: sugiere una diferencia real "
                "(la sesión 0.7 lo confirma)."],
        ["0.5", "H0: μ_deporte = μ_sin deporte. H1: μ_deporte ≠ μ_sin deporte. Es bilateral: la pregunta dice «distinto», no «menor»."],
        ["0.6", "1 − 0.95¹⁰ = 0.40. Hay 40% de probabilidad de al menos una falsa alarma."],
        ["0.7", "Numérica y 2 grupos → t de Welch. Adulto 200.8 vs Adulto Mayor 207.5: t = −1.29, p = 0.20 → "
                "no hay evidencia de diferencia."],
    ], [16 * mm, CONTENT_W - 16 * mm]))
    s.append(Spacer(1, 7))

    left = [box([P("<b>Criterio de dominio: paso a la Clase 17 si…</b>", "cellb"), Spacer(1, 3),
                 bullets(["Explico sin apuntes qué es un p-valor y qué es un IC 95%",
                          "Distingo desviación estándar de error estándar",
                          "Ante una pregunta nueva, elijo la prueba correcta",
                          "La ejecuto en Python y la concluyo en una frase con unidades",
                          "Respondo 5 de 6 en la evaluación del domingo 18"], "cell", GREEN, SYM)],
                bg=GREEN_BG, border=GREEN, width=(CONTENT_W - 6 * mm) * 0.5)]
    right = [P("Recursos externos (solo si algo no queda claro)", "h3"),
             bullets(["<b>StatQuest (Josh Starmer), YouTube</b>: videos de 10–15 min sobre p-valores, IC y TCL. "
                      "Es el mejor recurso para pasar de fórmula a intuición. En inglés, con subtítulos.",
                      "<b>OpenIntro Statistics (libro gratuito)</b>, capítulos 5–7: los mismos temas con ejercicios. "
                      "Las soluciones de los ejercicios impares están al final del libro.",
                      "<b>Seeing Theory (Brown University)</b>: visualizaciones interactivas del TCL y de los IC. "
                      "Sirve para jugar con la figura de la sesión 0.4."], "cell", E0)]
    s.append(two_cols(left, right))
    s.append(PageBreak())

    # ------------------------------------------------------------ evaluaciones
    s.append(band("Evaluaciones de los domingos", "Responde por escrito y sin apuntes. Las respuestas están abajo: míralas solo al terminar.", colors.HexColor("#059669")))
    s.append(Spacer(1, 7))
    dom11 = [
        "La variable «nivel educacional» (básica, media, superior), ¿es numérica, nominal u ordinal?",
        "Una variable tiene media 80 y mediana 60. ¿Hacia qué lado está sesgada?",
        "Un valor tiene z = −2. ¿Es un valor común o raro? ¿Por qué?",
        "¿Qué mide la desviación estándar y qué mide el error estándar?",
        "Si n pasa de 100 a 400, ¿qué le pasa al error estándar?",
        "¿El Teorema Central del Límite dice que, con n grande, los datos son normales?",
    ]
    dom18 = [
        "El IC 95% de una diferencia de medias es [−2; 8]. ¿Hay diferencia significativa con α = 0.05?",
        "Una prueba da p = 0.20. Escribe la conclusión correcta en una frase.",
        "Quieres comparar el ingreso medio de 4 regiones. ¿Qué prueba usas?",
        "¿El género y la preferencia de producto (A, B o C) están asociados? ¿Qué prueba usas?",
        "Un estudio con 1 millón de personas halla una diferencia de 0.1 mg/dL con p < 0.001. ¿Es relevante?",
        "¿Qué es la potencia de una prueba y cómo la aumentas?",
    ]
    half = (CONTENT_W - 6 * mm) * 0.5
    def qlist(title, qs):
        return [P(title, "h3"), numbered([("", q) for q in qs], E0)]
    s.append(two_cols(qlist("Domingo 11 oct · sesiones 0.1 a 0.3", dom11),
                      qlist("Domingo 18 oct · toda la etapa", dom18)))
    s.append(Spacer(1, 10))
    s.append(P("Respuestas", "h2"))
    r11 = ["Categórica ordinal: tiene orden, pero no distancias numéricas.",
           "A la derecha: la media es arrastrada por valores altos.",
           "Raro: solo ~2.5% de los valores está por debajo de −2σ.",
           "s: cuánto varían las personas. EE: cuán preciso es el promedio.",
           "Se reduce a la mitad (√400 / √100 = 2).",
           "No. Habla de la distribución de la MEDIA, no de los datos."]
    r18 = ["No: el intervalo contiene el 0.",
           "No hay evidencia suficiente para rechazar H0 (no significa que H0 sea cierta).",
           "ANOVA (Clase 17). Varias pruebas t inflarían el error tipo I.",
           "Chi-cuadrado de independencia: dos variables categóricas.",
           "Es significativa, pero sin importancia práctica: el efecto es mínimo.",
           "La probabilidad de detectar un efecto real. Se aumenta con más n (o con un efecto mayor o un α mayor)."]
    s.append(table([["#", "Domingo 11", "Domingo 18"]] +
                   [[f"P{i}", a, b] for i, (a, b) in enumerate(zip(r11, r18), 1)],
                   [10 * mm, (CONTENT_W - 10 * mm) / 2, (CONTENT_W - 10 * mm) / 2]))
    s.append(Spacer(1, 6))
    s.append(P("Nota: 5 de 6 correctas el domingo 18 es parte del criterio de dominio. Si fallas una pregunta, "
               "vuelve a la sesión que la cubre antes de empezar la Clase 17.", "small"))
    build("01_Etapa0_Estadistica.pdf", s, "Etapa 0 · Nivelación estadística")


if __name__ == "__main__":
    etapa0_pdf()
    print("OK")
