"""Construye las guías visuales en PDF a partir de las figuras de img/.

Requiere reportlab. Ejecutar figuras.py antes.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"
OUT = HERE.parent

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Segoe", str(FONTS / "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("Segoe-B", str(FONTS / "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("Segoe-I", str(FONTS / "segoeuii.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFont(TTFont("Sym", str(FONTS / "seguisym.ttf")))
pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe-B",
                              italic="Segoe-I", boldItalic="Segoe-B")

INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
SOFT = colors.HexColor("#F3F4F6")
C17 = colors.HexColor("#2563EB")
C18 = colors.HexColor("#0D9488")
C19 = colors.HexColor("#7C3AED")
NAVY = colors.HexColor("#0F172A")
AMBER = colors.HexColor("#D97706")
AMBER_BG = colors.HexColor("#FFFBEB")
RED = colors.HexColor("#DC2626")
RED_BG = colors.HexColor("#FEF2F2")
GREEN = colors.HexColor("#059669")
GREEN_BG = colors.HexColor("#ECFDF5")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def st(name, **kw):
    base = dict(fontName="Segoe", fontSize=9.6, leading=14.2, textColor=INK,
                alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "h1": st("h1", fontName="Segoe-B", fontSize=24, leading=28, textColor=NAVY),
    "sub": st("sub", fontSize=11.5, leading=15, textColor=MUTED),
    "h2": st("h2", fontName="Segoe-B", fontSize=14, leading=18, textColor=NAVY,
             spaceBefore=6, spaceAfter=4, keepWithNext=1),
    "h3": st("h3", fontName="Segoe-B", fontSize=10.8, leading=14, textColor=NAVY,
             spaceBefore=5, spaceAfter=2, keepWithNext=1),
    "body": st("body"),
    "small": st("small", fontSize=8.4, leading=11, textColor=MUTED),
    "cell": st("cell", fontSize=8.8, leading=11.6),
    "cellb": st("cellb", fontName="Segoe-B", fontSize=8.8, leading=11.6),
    "mono": st("mono", fontName="Mono", fontSize=8.6, leading=11.4),
    "formula": st("formula", fontSize=9.6, leading=17),
    "big": st("big", fontName="Segoe-B", fontSize=12, leading=16.5, textColor=NAVY),
    "band": st("band", fontName="Segoe-B", fontSize=16, leading=20, textColor=colors.white),
    "bandsub": st("bandsub", fontSize=9.5, leading=12.5, textColor=colors.white),
}


def P(text, style="body"):
    if style == "mono":
        stripped = text.lstrip(" ")
        text = "&nbsp;" * (len(text) - len(stripped)) + stripped
    return Paragraph(text, S[style])


def bullets(items, style="body", color=INK, marker="•"):
    rows = [[P(f'<font color="{color.hexval().replace("0x", "#")}"><b>{marker}</b></font>', style),
             P(t, style)] for t in items]
    t = Table(rows, colWidths=[5 * mm, None])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    return t


def numbered(items, color):
    rows = []
    for i, (title, text) in enumerate(items, 1):
        badge = Table([[P(f'<font color="white"><b>{i}</b></font>', "cell")]],
                      colWidths=[6 * mm], rowHeights=[6 * mm])
        badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                                   ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                                   ("ROUNDEDCORNERS", [3, 3, 3, 3])]))
        rows.append([badge, P(f"<b>{title}.</b> {text}" if title else text)])
    t = Table(rows, colWidths=[9 * mm, None])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def box(flowables, bg=SOFT, border=None, width=CONTENT_W, pad=7):
    t = Table([[flowables]], colWidths=[width])
    style = [("BACKGROUND", (0, 0), (-1, -1), bg),
             ("LEFTPADDING", (0, 0), (-1, -1), pad + 2),
             ("RIGHTPADDING", (0, 0), (-1, -1), pad + 2),
             ("TOPPADDING", (0, 0), (-1, -1), pad),
             ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
             ("ROUNDEDCORNERS", [5, 5, 5, 5])]
    if border is not None:
        style.append(("LINEBEFORE", (0, 0), (0, -1), 3, border))
    t.setStyle(TableStyle(style))
    return t


def band(title, subtitle, color):
    t = Table([[[P(title, "band"), Spacer(1, 2), P(subtitle, "bandsub")]]],
              colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                           ("ROUNDEDCORNERS", [6, 6, 6, 6])]))
    return t


def two_cols(left, right, gap=6 * mm, ratio=0.5):
    lw = (CONTENT_W - gap) * ratio
    rw = CONTENT_W - gap - lw
    t = Table([[left, "", right]], colWidths=[lw, gap, rw])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def img(name, width=CONTENT_W):
    from reportlab.lib.utils import ImageReader
    path = IMG / name
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def table(rows, widths, header_color=NAVY, zebra=True):
    data = [[P(c, "cellb" if i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    data[0] = [P(f'<font color="white">{c}</font>', "cellb") for c in rows[0]]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), header_color),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
             ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if zebra:
        for i in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    t.setStyle(TableStyle(style))
    return t


def footer_factory(label):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Segoe", 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm, label)
        canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"{doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        canvas.restoreState()
    return footer


def build(filename, story, label):
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=18 * mm,
                            title=label, author="Guía de estudio")
    f = footer_factory(label)
    doc.build(story, onFirstPage=f, onLaterPages=f)


# ============================================================== 00 ROADMAP
def roadmap_pdf():
    s = []
    s.append(P("Roadmap de estudio", "h1"))
    s.append(Spacer(1, 3))
    s.append(P("Diplomado Data Science · desde la Clase 17 hasta ponerse al día · versión 1 · 5 oct 2026", "sub"))
    s.append(Spacer(1, 10))

    stats = [("23", "clases de atraso", "C17 → C40"), ("8 h", "por semana", "5 sesiones de ~95 min"),
             ("~18", "semanas", "para cubrir C17–35"), ("1", "entrega cercana", "Proyecto N°2 · ~5 nov")]
    cells = []
    for big, label, sub in stats:
        cells.append([P(f'<font size="20" color="#0F172A"><b>{big}</b></font>', "body"),
                      Spacer(1, 6), P(f"<b>{label}</b>", "cell"), P(sub, "small")])
    w = (CONTENT_W - 3 * 4 * mm) / 4
    t = Table([[cells[0], "", cells[1], "", cells[2], "", cells[3]]],
              colWidths=[w, 4 * mm, w, 4 * mm, w, 4 * mm, w])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), SOFT), ("BACKGROUND", (2, 0), (2, 0), SOFT),
                           ("BACKGROUND", (4, 0), (4, 0), SOFT), ("BACKGROUND", (6, 0), (6, 0), SOFT),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("ROUNDEDCORNERS", [5, 5, 5, 5])]))
    s.append(t)
    s.append(Spacer(1, 12))

    s.append(P("Las etapas en el tiempo", "h2"))
    s.append(img("roadmap.png", width=CONTENT_W * 0.9))
    s.append(P("La letra dentro de cada barra es el nivel de profundidad. "
               "<b>A</b> = a fondo (es base de lo que viene). <b>B</b> = práctico (aplicar e interpretar). "
               "La Etapa 0 no tiene carpeta: es una nivelación de estadística, porque las Clases 17–24 dependen de ella.",
               "small"))
    s.append(Spacer(1, 4))

    dias = [("Lun", "Nuevo", C17), ("Mar", "Nuevo", C17), ("Mié", "Nuevo", C17), ("Jue", "Libre", LINE),
            ("Vie", "Nuevo", C17), ("Sáb", "Libre", LINE), ("Dom", "Repaso", GREEN)]
    row = []
    for d, uso, c in dias:
        fg = "white" if c not in (LINE,) else "#6B7280"
        row.append([P(f'<font color="{fg}"><b>{d}</b></font>', "cell"),
                    P(f'<font color="{fg}">{uso}</font>', "small")])
    dw = (CONTENT_W * 0.52) / 7
    week = Table([row], colWidths=[dw] * 7)
    ws = [("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    for i, (_, _, c) in enumerate(dias):
        ws.append(("BACKGROUND", (i, 0), (i, 0), c))
        ws.append(("LINEAFTER", (i, 0), (i, 0), 2, colors.white))
    week.setStyle(TableStyle(ws))

    sesion = table([["Min", "Bloque de cada sesión"],
                    ["10", "Repaso: 3 preguntas de la sesión anterior, sin apuntes"],
                    ["25", "Concepto nuevo: transcripción + guía"],
                    ["45", "Práctica guiada en notebook (Python)"],
                    ["15", "Ejercicio de cierre sin ayuda + notas propias"]],
                   [12 * mm, CONTENT_W * 0.52 - 12 * mm])

    left = [P("Ritmo semanal", "h2"), week, Spacer(1, 8), sesion]
    right = [P("Reglas del plan", "h2"),
             bullets([
                 "<b>Se avanza por dominio, no por haber leído.</b> Cada etapa tiene un criterio concreto en su guía.",
                 "<b>Domingo = repaso espaciado:</b> la semana actual, y además la etapa cerrada hace 1 y 3 semanas.",
                 "<b>R → Python:</b> las Clases 17–27 usan R. Se lee el código R y se practica en Python con <i>statsmodels</i>.",
                 "<b>Transcripción primero:</b> es la fuente principal; PDF y notebook la complementan.",
             ])]
    s.append(two_cols(left, right, ratio=0.54))
    s.append(Spacer(1, 10))

    s.append(P("Material que falta generar", "h2"))
    s.append(table([["Se necesita en", "Falta"],
                    ["Semana 9 · 30 nov", "Transcripción Clase 24"],
                    ["Semana 11 · 14 dic", "Transcripciones Clases 25–27 · PDF «Clase 3» de series de tiempo"],
                    ["Semana 13 · 28 dic", "Transcripciones Clases 30, 31 y 32"],
                    ["Semana 16 · 18 ene", "Transcripción Clase 33 (existe el .mp4)"],
                    ["Etapa 8", "Carpetas Clases 28, 29 y 36–40"]],
                   [40 * mm, CONTENT_W - 40 * mm]))
    build("00_Roadmap.pdf", s, "Roadmap de estudio · Diplomado Data Science")


# ===================================================== 01 REGRESIÓN C17-19
def class_page(s, color, num, titulo, idea, conceptos, formulas, salida, figura, pie,
               cuidado, py):
    s.append(band(f"Clase {num} · {titulo}", "Fuente: transcripción de la clase + notebook R", color))
    s.append(Spacer(1, 8))
    s.append(box([P(idea, "big")], bg=SOFT, border=color))
    s.append(Spacer(1, 8))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, color))
    s.append(Spacer(1, 6))
    s.append(img(figura))
    s.append(P(pie, "small"))
    s.append(Spacer(1, 6))
    left = [P("Fórmulas", "h3"),
            box([P(f, "formula") for f in formulas], bg=SOFT),
            Spacer(1, 6),
            P("En Python", "h3"),
            box([P(f, "mono") for f in py], bg=SOFT)]
    right = [P("Cómo leer la salida", "h3"), bullets(salida, "cell", color),
             Spacer(1, 6),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado],
                 bg=AMBER_BG, border=AMBER, width=(CONTENT_W - 6 * mm) * 0.5)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


def regresion_pdf():
    s = []
    # ---------------------------------------------------------- overview
    s.append(band("Carpeta 1 · Análisis de regresión",
                  "Clases 17, 18 y 19 · Profesora Ana María Alvarado · Etapa 1 del roadmap (semanas 3–5)",
                  NAVY))
    s.append(Spacer(1, 10))
    s.append(box([P("La gran idea de la carpeta", "cellb"), Spacer(1, 3),
                  P("ANOVA y regresión son <b>el mismo modelo</b>:  <b>Y = parte explicada + error</b>. "
                    "ANOVA explica Y con variables <b>categóricas</b> (grupos), la regresión con variables "
                    "<b>numéricas</b>, y la regresión múltiple mezcla ambas usando <b>variables dummy</b>. "
                    "Si entiendes esto, las tres clases son una sola historia.", "body")],
                 bg=colors.HexColor("#EFF6FF"), border=C17))
    s.append(Spacer(1, 8))
    s.append(P("Mapa conceptual", "h2"))
    s.append(img("mapa_c17_19.png"))
    s.append(Spacer(1, 6))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Media, varianza y desviación estándar",
                     "Distribución normal",
                     "Prueba de hipótesis: H0 / H1, p-valor y α",
                     "Intervalo de confianza",
                     "Cuartiles y boxplot",
                     "→ Todo esto es la <b>Etapa 0</b> del roadmap"], "cell", C17)]
    right = [P("Hacia dónde lleva", "h3"),
             bullets(["<b>GLM (C20–21):</b> la regresión logística es la regresión cuando Y es categórica",
                      "<b>A/B testing (C22):</b> mismo marco de prueba de hipótesis que ANOVA",
                      "<b>Machine Learning (C30+):</b> árboles y random forest cuando la regresión no alcanza",
                      "<b>Regularización:</b> LASSO / Ridge como alternativa a stepwise"], "cell", C17)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 6))
    s.append(table([["Dataset", "Clase", "Pregunta que responde"],
                    ["salarios.xlsx (40 filas)", "17 · 18 · 19", "¿Cambia el salario según industria, sexo y experiencia?"],
                    ["biomass.txt (45 filas, 17 col.)", "18 · 19", "¿Qué propiedades del suelo predicen la biomasa?"],
                    ["bio_new.csv", "19", "Predicción con datos nuevos (quedó pendiente en clase)"],
                    ["ENS.csv", "17", "Datos de apoyo del notebook"]],
                   [48 * mm, 24 * mm, CONTENT_W - 72 * mm]))
    s.append(PageBreak())

    # ---------------------------------------------------------- clase 17
    class_page(
        s, C17, 17, "ANOVA: comparar medias de grupos",
        "¿Las medias de varios grupos son realmente distintas, o la diferencia es puro ruido? "
        "ANOVA compara la variación ENTRE grupos con la variación DENTRO de los grupos.",
        [("Modelo básico", "Todo dato = un promedio + un error aleatorio (Y = μ + ε). Es el punto de partida de todos los modelos."),
         ("ANOVA de 1 factor", "H0: todas las medias son iguales. H1: al menos dos son distintas. Si F es grande, los grupos difieren más de lo que explica el azar."),
         ("Factor en R", "Una categórica codificada con números (región 1–16) debe declararse <i>factor</i>; si no, el modelo la trata como cantidad y el resultado no tiene sentido."),
         ("Comparaciones múltiples", "ANOVA dice SI hay diferencias, no CUÁLES. Tukey (grupos de tamaño similar) o Scheffé (tamaños desiguales) comparan par a par."),
         ("Interacción", "El efecto de un factor puede depender del otro. En los datos de la clase, las mujeres ganan menos en retail y banca, pero más en minería.")],
        ["Y_ij = μ + α_j + ε_ij",
         "Y = μ + α_j + β_k + (αβ)_jk + ε",
         "SC_total = SC_entre + SC_dentro",
         "F = CM_entre / CM_dentro"],
        ["<b>Pr(&gt;F) &lt; 0.05</b> → hay diferencias entre grupos",
         "Coeficientes: la 1ª categoría es la base; las demás son diferencias respecto a ella",
         "IC de dos grupos que no se solapan → diferencia significativa",
         "Estrellas: *** p≈0 · ** p&lt;0.01 · * p&lt;0.05"],
        "c17.png",
        "Datos reales de salarios.xlsx. Izquierda: la variación entre industrias es visible. "
        "Derecha: las líneas se cruzan en minería → hay interacción industria × sexo.",
        ["No elimines outliers «por receta»: entiende primero por qué son atípicos.",
         "Supuestos: errores independientes, normales y con varianza constante."],
        ["smf.ols('Salario ~ C(Industria)', d).fit()",
         "sm.stats.anova_lm(modelo)",
         "pairwise_tukeyhsd(d.Salario, d.Industria)"])

    # ---------------------------------------------------------- clase 18
    class_page(
        s, C18, 18, "Correlación y regresión lineal simple",
        "La correlación mide si dos variables se mueven juntas en línea recta. "
        "La regresión convierte esa relación en una ecuación para predecir.",
        [("Correlación de Pearson", "Va de −1 a 1 y mide relación LINEAL. r ≈ 0 no significa «sin relación»: puede haber una relación curva (ejemplo del datasaurus)."),
         ("Correlación ≠ causalidad", "Que dos variables se muevan juntas no prueba que una cause la otra."),
         ("Regresión simple", "Y = β0 + β1·X + ε, estimada por mínimos cuadrados. β1 = cuánto cambia Y por cada unidad de X."),
         ("R²", "% de la variabilidad de Y que explica el modelo. En regresión simple, R² = r²."),
         ("Diagnóstico de residuos", "Normalidad (QQ-plot, Shapiro-Wilk), independencia (Durbin-Watson), homocedasticidad (Breusch-Pagan). Aquí queremos NO rechazar H0.")],
        ["Y = β0 + β1·X + ε",
         "R² = SC_regresión / SC_total",
         "R² = r²   (solo regresión simple)"],
        ["<b>Estimate</b>: los β. <b>Pr(&gt;|t|)</b>: prueba H0: β = 0",
         "Si β1 no es significativo, X no aporta",
         "<b>R²</b> cerca de 1 = buen ajuste (pero un R² = 1 es sospechoso)",
         "No extrapolar fuera del rango de X observado"],
        "c18.png",
        "Datos reales de biomass.txt: r = 0.77 → R² = 0.77² = 0.60. El pH explica el 60% de la variación de la biomasa. "
        "El intercepto negativo (−885) no tiene sentido físico: pH = 0 está fuera del rango de los datos.",
        ["Un outlier en X puede «tirar» la recta (efecto palanca). Antes de borrarlo, márcalo con una dummy y prueba si influye.",
         "No descartes una variable por baja correlación: prueba Spearman o un modelo múltiple."],
        ["stats.pearsonr(x, y)",
         "m = smf.ols('BIO ~ Ph', d).fit()",
         "m.summary()",
         "het_breuschpagan(m.resid, m.model.exog)"])

    # ---------------------------------------------------------- clase 19
    class_page(
        s, C19, 19, "Regresión lineal múltiple",
        "Con varios predictores, cada coeficiente dice el efecto de una variable MANTENIENDO LAS DEMÁS "
        "CONSTANTES. Y hay que vigilar que los predictores no se pisen entre sí.",
        [("Lineal = en los parámetros", "Y = β0 + β1·X² sigue siendo lineal (X² es otra variable). Lo que no puede haber es β al cuadrado."),
         ("Dos pruebas", "t: ¿esta variable aporta? (H0: βj = 0). F global: ¿el modelo aporta algo? (H0: todos los β = 0)."),
         ("Variables dummy", "Una categórica con m niveles usa m − 1 dummies. Usar las m junto al intercepto hace la matriz no invertible (dummy trap)."),
         ("R² ajustado, AIC y BIC", "El R² SIEMPRE sube al agregar variables, aunque no sirvan. Para comparar modelos de distinto tamaño usa R² ajustado, AIC o BIC."),
         ("Multicolinealidad y stepwise", "Predictores muy correlacionados inflan el VIF y vuelven inestables los β. Forward, backward y both pueden terminar en modelos distintos.")],
        ["Y = β0 + β1·X1 + … + βk·Xk + ε",
         "VIF_j = 1 / (1 − R²_j)",
         "N° de subconjuntos de k variables = 2^k"],
        ["Cada β: efecto de Xj con las demás fijas",
         "VIF &lt; 5 OK · 5–10 revisar · &gt; 10 problema",
         "Menor AIC/BIC = mejor equilibrio ajuste/simplicidad",
         "R² alto + residuos malos = el modelo <b>NO</b> sirve todavía"],
        "c19.png",
        "VIF calculado sobre biomass.txt. El pH, el mejor predictor individual de la Clase 18, tiene VIF = 62: "
        "está muy correlacionado con BUF, Mg y Ca. Por eso su coeficiente se vuelve inestable en el modelo completo.",
        ["Conclusión de la profesora: el modelo elegido (R² ≈ 0.85) cumple normalidad y homocedasticidad, pero falla en independencia (Durbin-Watson), así que todavía no sirve. "
         "El siguiente paso sería transformar (Box-Cox) o cambiar de modelo.",
         "Con 16 candidatas hay 2^16 = 65.536 modelos posibles (no 16!)."],
        ["smf.ols('BIO ~ Ph + SAL + k', d).fit()",
         "variance_inflation_factor(X, j)",
         "m.aic, m.bic, m.rsquared_adj",
         "pd.get_dummies(d, drop_first=True)"])

    # ---------------------------------------------------------- R -> Python
    s.append(band("Traductor R → Python", "Para leer los notebooks de clase y practicar en Python", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["En la clase (R)", "En Python", "Para qué"],
                    ["skim(d) / summary(d)", "d.describe(include='all')", "Explorar los datos"],
                    ["factor(x)", "C(x) en la fórmula · pd.Categorical", "Tratar x como grupo"],
                    ["aov(y ~ x)", "smf.ols('y ~ C(x)', d).fit() + anova_lm()", "ANOVA"],
                    ["TukeyHSD()", "pairwise_tukeyhsd()", "Comparaciones múltiples"],
                    ["cor() / cor.test()", "d.corr() · stats.pearsonr()", "Correlación"],
                    ["lm(y ~ x)", "smf.ols('y ~ x', d).fit()", "Regresión"],
                    ["summary(m)", "m.summary()", "Tabla de resultados"],
                    ["shapiro.test()", "stats.shapiro(m.resid)", "Normalidad de residuos"],
                    ["dwtest()", "durbin_watson(m.resid)", "Independencia"],
                    ["bptest()", "het_breuschpagan(m.resid, m.model.exog)", "Homocedasticidad"],
                    ["vif()", "variance_inflation_factor()", "Multicolinealidad"],
                    ["AIC(m) / BIC(m)", "m.aic / m.bic", "Comparar modelos"],
                    ["step()", "No existe directo: se programa comparando AIC", "Selección stepwise"],
                    ["predict(m, nuevos)", "m.predict(nuevos)", "Predecir"]],
                   [45 * mm, 78 * mm, CONTENT_W - 123 * mm]))
    s.append(Spacer(1, 6))
    s.append(P("Imports: <font name='Mono'>import statsmodels.formula.api as smf · import statsmodels.api as sm · from scipy import stats</font>", "small"))
    s.append(Spacer(1, 12))
    s.append(P("Checklist: ¿mi regresión es confiable?", "h2"))
    checks = [
        ("Linealidad", "Gráfico Y vs X y residuos vs ajustados sin forma de curva"),
        ("Normalidad de residuos", "QQ-plot sobre la diagonal · Shapiro-Wilk p &gt; 0.05"),
        ("Independencia", "Durbin-Watson cerca de 2"),
        ("Homocedasticidad", "Nube de residuos sin forma de embudo · Breusch-Pagan p &gt; 0.05"),
        ("Multicolinealidad", "Todos los VIF &lt; 5 (o justificar los que están entre 5 y 10)"),
        ("Puntos influyentes", "Revisar outliers / palanca antes de decidir qué hacer"),
        ("Rango", "Predecir solo dentro del rango de X observado"),
    ]
    s.append(table([["", "Supuesto", "Cómo lo verifico"]] + [['<font name="Sym">☐</font>', a, b] for a, b in checks],
                   [8 * mm, 42 * mm, CONTENT_W - 50 * mm]))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Ojo con el p-valor en los diagnósticos.</b> En las pruebas de supuestos, H0 es «el supuesto "
                    "se cumple». Por eso aquí queremos un p-valor GRANDE (&gt; 0.05). Es al revés que al probar "
                    "si un coeficiente es significativo.", "cell")], bg=AMBER_BG, border=AMBER))
    s.append(PageBreak())

    # ---------------------------------------------------------- alertas + repaso
    s.append(band("Alertas del material, repaso y criterio de dominio", "Lo que hay que corregir, lo que debes saber responder y cuándo pasar a la Clase 20", NAVY))
    s.append(Spacer(1, 8))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Definición imprecisa del p-valor (Clase 17)</b>", "cellb"), Spacer(1, 2),
                  P("En la transcripción se dice que «el valor p es la probabilidad de rechazar H0 cuando H0 es correcta». "
                    "Eso es <b>α</b> (el nivel de significancia, o probabilidad de error tipo I), que se fija ANTES del análisis.", "cell"),
                  Spacer(1, 3),
                  P("<b>Definición correcta:</b> el p-valor es la probabilidad de obtener un resultado <b>igual o más extremo</b> "
                    "que el observado, <b>suponiendo que H0 es verdadera</b>. Se compara con α: si p &lt; α, se rechaza H0.", "cell"),
                  Spacer(1, 3),
                  P("En la misma frase se dice «siempre quiero rechazar H0». Eso no aplica a los diagnósticos de "
                    "supuestos, donde queremos NO rechazar.", "cell")],
                 bg=RED_BG, border=RED))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Otras observaciones</b>", "cellb"),
                  bullets(["La transcripción de la Clase 18 tiene un tramo ilegible por error del reconocimiento de voz. No es un error de la profesora.",
                           "En la Clase 19 se duda entre 16! y 2^16 posibles modelos. Lo correcto es 2^16 = 65.536: cada variable está o no está.",
                           "<b>bio_new.csv tiene un error de datos:</b> las columnas Mn, Zn y Cu tienen el punto decimal corrido (ej. Cu 502.381 vs 5.02381 en biomass). Un modelo que usa esas variables predice biomasas de ~100.000. Lo trabajas en la Parte D del notebook.",
                           "Esta guía se basa en las transcripciones y los notebooks. Las diapositivas PDF no se revisaron página a página."],
                          "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 8))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    preguntas = [
        ("Un modelo con 10 variables tiene R² = 0.95 y otro con 3 tiene R² = 0.70. ¿El primero es mejor?",
         "No necesariamente. El R² siempre sube al agregar variables. Compara con R² ajustado, AIC o BIC."),
        ("El p-valor de β1 es 0.6. ¿Qué concluyes?",
         "No se rechaza H0: β1 = 0. No hay evidencia de que esa variable aporte al modelo."),
        ("Breusch-Pagan da p = 0.98. ¿Es buena o mala noticia?",
         "Buena. H0 es «hay homocedasticidad» y no se rechaza, así que el supuesto se cumple."),
        ("Dos predictores tienen correlación 0.95 entre sí. ¿Los incluyes juntos?",
         "Mejor no. Hay multicolinealidad: el VIF sube y los β se vuelven inestables. Quédate con uno o combínalos."),
        ("Tu modelo tiene buen R² pero los residuos fallan independencia. ¿Lo entregas?",
         "No. Revisa puntos influyentes, prueba transformaciones (Box-Cox) o cambia de tipo de modelo."),
        ("En el gráfico de interacción, ¿qué significa que las líneas se crucen?",
         "Que el efecto de un factor depende del nivel del otro. No basta un modelo aditivo."),
    ]
    rows = [["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preguntas, 1)]
    s.append(table(rows, [7 * mm, 78 * mm, CONTENT_W - 85 * mm]))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 20</b>", "cellb"), Spacer(1, 3),
                  bullets(["Explico sin apuntes qué compara ANOVA y qué dice un p-valor",
                           "Ajusto en Python una regresión múltiple con una variable categórica",
                           "Interpreto coeficientes, p-valores, R² ajustado y AIC",
                           "Verifico los 4 supuestos y decido si el modelo sirve",
                           "Predigo con bio_new.csv y detecto el error de sus datos"], "cell", GREEN, '<font name="Sym">☐</font>')],
                 bg=GREEN_BG, border=GREEN))
    build("02_Regresion_C17-19.pdf", s, "Carpeta 1 · Análisis de regresión · Clases 17–19")


if __name__ == "__main__":
    roadmap_pdf()
    regresion_pdf()
    print("OK")
