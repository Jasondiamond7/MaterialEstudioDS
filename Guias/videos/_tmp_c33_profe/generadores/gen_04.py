import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, alerta

OUT = out("guion_c33_04_motivos.json")
v = V("Irrelevante, redundante o útil en conjunto")
CF, SO, FA, PA, MP = D["corr_fuga"], D["sospechosas"], D["fallas"], D["pares90"], D["mi_par"]

v.portada("¿Por qué sobra una variable?", "Irrelevante, redundante o útil solo en conjunto: tres motivos, tres decisiones",
          "Tercera parada: los tres motivos por los que una variable puede sobrar, o parecer que sobra. Y cada uno pide una decisión distinta.")

# ---- 1. Hook: la banda
v.b("Piensa en una banda de música. [[a]]Un músico que no toca nada: sobra, se va. [[b]]Dos guitarristas tocando exactamente las mismas notas: con uno basta. [[c]]Y el bajista: solo, suena raro. Pero con la banda, sostiene todo.",
    CL(), P("una banda de música", 960, 210, "violeta", 54, ent="subir"),
    tarjetas([(360, "gris", "no toca nada", "sobra: se va", "a"), (960, "azul", "dos guitarras iguales", "con una basta", "b"),
              (1560, "verde", "el bajista", "solo suena raro · con la banda sostiene todo", "c")], y=540, h=380),
    ilustrativo(1000, texto="analogía"), sec="1. Tres motivos, tres decisiones")
v.b("Con las variables pasa lo mismo. [[a]]Irrelevante: no aporta, ni sola ni acompañada. Se descarta. [[b]]Redundante: repite la información de otra. Se conserva una. [[c]]Y útil solo en conjunto: sola parece ruido, acompañada aporta. Los filtros univariados la matan.",
    CL(), tarjetas([(360, "gris", "irrelevante", "ni sola ni acompañada → se descarta", "a"), (960, "azul", "redundante", "repite a otra → queda una", "b"),
                    (1560, "verde", "útil en conjunto", "sola parece ruido → NO la descartes", "c")], y=520, h=400),
    P("los filtros univariados la matan", 1560, 820, "rojo", 36, en="c", t=0.8))

# ---- 2. Correlación con la fuga
ORD = list(CF.items())


def barra_corr(i, k, t):
    nm, r = ORD[i]
    y = 270 + 52 * i
    c = "rojo" if r > 0 else "azul"
    w = abs(r) * 900
    x0, x1 = (1060, 1060 + w) if r > 0 else (1060 - w, 1060)
    return [T(nm, 1040 if r > 0 else 1080, y, tam=28, c="tinta", w="m", anc="rm" if r > 0 else "lm", en=k, t=t),
            Bx(x0, y - 17, x1, y + 17, c, None, 0, 5, grow="x", en=k, t=t + 0.05),
            T(f"{r:+.3f}".replace("-", "−"), (x1 + 14) if r > 0 else (x0 - 14), y, tam=26, c=c, w="sb", anc="lm" if r > 0 else "rm", en=k, t=t + 0.3)]


v.b("La herramienta básica es la correlación de cada variable original con la fuga. [[a]]Reclamos es la más fuerte, con cero coma cincuenta y uno. [[b]]Estado dos, cero coma treinta y dos. Llamadas, segundos y valor del cliente, alrededor de menos cero coma veintinueve.",
    CL(), T("correlación de Pearson con la fuga · base limpia", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    L(1060, 240, 1060, 870, "tinta", 3), barra_corr(0, "a", 0), [barra_corr(i, "b", 0.25 * (i - 1)) for i in range(1, 5)],
    sec="2. La correlación con la fuga")
v.b("Y al final de la lista, [[a]]tres quedan prácticamente en cero: antigüedad, edad, y número de fallas, con menos cero coma cero cero nueve. [[b]]¿Son irrelevantes?",
    [barra_corr(i, None, 0.12 * (i - 5)) for i in range(5, 9)], [barra_corr(i, "a", 0.3 * (i - 9)) for i in range(9, 12)],
    R(1060, 270 + 52 * 10, 900, 170, "ambar", 26, en="a", t=1.2), P("¿irrelevantes?", 1560, 720, "ambar", 48, en="b"))
v.b("Antes de descartarlas, una segunda opinión. [[a]]Pearson solo mide relación lineal, así que miramos también la información mutua, que capta cualquier dependencia. [[b]]Número de fallas: información mutua cero, y p-valor cero coma sesenta. Edad: cero, y p-valor cero coma treinta.",
    CL(), T("la segunda opinión · filtros sobre la base", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    T("variable", 420, 300, tam=32, c="gris", w="sb"), T("info. mutua", 900, 300, tam=32, c="gris", w="sb"), T("F de ANOVA", 1220, 300, tam=32, c="gris", w="sb"),
    T("p-valor", 1520, 300, tam=32, c="gris", w="sb"),
    *[[Bc(960, 400 + 110 * i, 1400, 90, "blanco", c, 3, 20, en=k, ent="subir", t=0.2 * (i % 2)), T(nm, 420, 400 + 110 * i, tam=34, c="tinta", w="m", en=k, t=0.2 * (i % 2) + 0.05),
       T(f"{SO[nm][0]:.4f}", 900, 400 + 110 * i, tam=38, c=c, en=k, t=0.2 * (i % 2) + 0.1), T(f"{SO[nm][1]:,.2f}".replace(",", "."), 1220, 400 + 110 * i, tam=38, c=c, en=k, t=0.2 * (i % 2) + 0.15),
       T(f"{SO[nm][2]:.3f}", 1520, 400 + 110 * i, tam=38, c=c, en=k, t=0.2 * (i % 2) + 0.2)]
      for i, (nm, c, k) in enumerate((("numero_fallas", "rojo", "b"), ("edad", "rojo", "b"), ("antiguedad", "ambar", None), ("reclamos", "verde", None)))],
    T("antigüedad: señal marginal · reclamos: señal fuerte, para comparar", 960, 870, tam=32, c="gris", w="sb", t=0.6))
v.b("Para un filtro univariado, número de fallas es ruido puro. [[a]]Correlación casi cero, información mutua cero, p-valor alto. [[b]]Cualquiera la descartaría. Pero hagamos la prueba correcta: medirla acompañada.",
    CL(), T("numero_fallas", 960, 280, tam=80, c="tinta", w="m", ent="subir"),
    *[P(tx, x, 450, "rojo", 44, en="a", t=0.35 * j) for j, (tx, x) in enumerate(((f"r = {CF['numero_fallas']:.3f}".replace("-", "−"), 520), (f"MI = {SO['numero_fallas'][0]:.0f}", 960), (f"p = {SO['numero_fallas'][2]:.2f}", 1400)))],
    T("veredicto del filtro: «ruido»", 960, 600, tam=56, c="gris", en="b", id="ver"), P("la prueba correcta: medirla acompañada", 960, 780, "verde", 50, en="b", t=1.0))

# ---- 3. El caso numero_fallas
AU = [("sola", FA["sola"], "rojo", "numero_fallas sola"), ("base", FA["base"], "gris", "base de 5"), ("base_f", FA["base_f"], "ambar", "base + fallas"),
      ("base_f_m", FA["base_f_m"], "verde", "base + fallas + monto")]


def barra_auc(i, en, t=0.0):
    _k, a, c, lab = AU[i]
    x = 420 + 360 * i
    h = (a - 0.45) * 1100
    return [Bx(x - 120, 860 - h, x + 120, 860, c, None, 0, 12, grow="y", en=en, t=t), N(a, x, 860 - h - 42, "{:.4f}", 52, c, en=en, t=t + 0.3, dur=0.8),
            T(lab, x, 905, tam=30, c=c, w="sb", en=en, t=t)]


v.b("La prueba de la clase usa regresión logística, con validación cruzada de cinco folds. [[a]]Número de fallas sola: AUC cero coma cinco dos seis. Prácticamente azar.",
    CL(), T("AUC · validación cruzada de 5 folds · base real", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    L(260, 860, 1660, 860, "tinta", 4), L(260, 860 - 0.05 * 1100, 1660, 860 - 0.05 * 1100, "gris", 3, guion=True), T("azar 0.5", 1760, 860 - 0.05 * 1100, tam=28, c="gris", w="sb"),
    barra_auc(0, "a"), T("eje desde 0.45", 1760, 900, tam=24, c="gris", w="sb"), sec="3. El caso numero_fallas")
v.b("Ahora, una base de cinco variables: reclamos, segundos, llamadas, estado dos y valor del cliente. [[a]]Da cero coma nueve cero seis nueve. [[b]]Le sumamos número de fallas: sube a cero coma nueve uno cero dos. [[c]]Y con monto, a cero coma nueve uno siete uno.",
    barra_auc(1, "a"), barra_auc(2, "b"), barra_auc(3, "c"))
v.b("Inútil sola, útil acompañada. [[a]]Y no es un ejemplo de libro: está en nuestros datos. [[b]]En el video ocho vas a ver que el Lasso le da a número de fallas el cuarto coeficiente más grande de las cincuenta candidatas.",
    CL(), P("inútil sola · útil acompañada", 960, 280, "verde", 64, ent="subir"),
    T("no es un ejemplo de libro: es la base real", 960, 450, tam=48, c="tinta", w="sb", en="a"),
    P(f"Lasso: numero_fallas = 4.º coeficiente (+{D['lasso_top'][3][1]:.3f})", 960, 640, "violeta", 50, en="b"), T("→ video 8", 960, 750, tam=36, c="gris", w="sb", en="b", t=0.6))
v.b("La frase que hay que llevarse es esta: [[a]]un p-valor alto no significa que la variable no sirve. [[b]]Significa que no sirve sola.",
    CL(), T("p = 0.60", 960, 300, tam=90, c="gris", ent="subir"), T("≠ «no sirve»", 960, 470, tam=80, c="rojo", en="a"),
    P("= «no sirve sola»", 960, 680, "verde", 80, en="b"))
v.b("Y aquí hay una alerta del material. [[a]]La diapositiva de los tres motivos pone a número de fallas y a la edad como ejemplos de irrelevantes. [[b]]Dos diapositivas después, la misma clase demuestra que número de fallas no lo es. [[c]]Los apuntes lo dicen mejor: en esta base no hay una irrelevante clara.",
    CL(), alerta(),
    T("«Irrelevante: numero_fallas, edad»", 960, 360, tam=56, c="gris", en="a", id="qirr"), U("qirr", tachado=1.0, en="b", dur=0.6),
    P(f"numero_fallas sube el AUC: {FA['base']:.3f} → {FA['base_f']:.3f}", 960, 540, "rojo", 46, en="b", t=0.3),
    P("irrelevante clara en esta base: ninguna", 960, 720, "verde", 50, en="c"))

# ---- 4. Redundancia
v.b("El segundo motivo es la redundancia. [[a]]Para encontrarla, se mira la correlación entre las candidatas. [[b]]Pero la matriz es simétrica, y tiene unos en la diagonal: dos de cada tres celdas sobran.",
    CL(), T("matriz de correlación entre candidatas", 960, 190, tam=40, c="gris", w="sb", ent="subir", en="a"),
    *[[Bc(780 + 150 * j, 370 + 150 * i, 140, 140, "blanco" if j > i else ("grisclaro" if i == j else "linea"), "teal" if j > i else None, 3, 12, en="a", t=0.06 * (3 * i + j), id=f"m{i}{j}"),
       T("1" if i == j else ("r" + "abc"[i] + "abc"[j] if j > i else "·"), 780 + 150 * j, 370 + 150 * i, tam=40, c="teal" if j > i else "gris", w="m", en="a", t=0.06 * (3 * i + j) + 0.05)]
      for i in range(3) for j in range(3)],
    T("diagonal = 1", 470, 520, tam=36, c="gris", w="sb", en="b"), P("cada par, una sola vez: triángulo superior", 960, 860, "teal", 46, en="b", t=0.6),
    sec="4. La redundancia")
v.b("Con tres variables quedan tres pares. [[a]]Con cincuenta, quedan cincuenta por cuarenta y nueve, dividido en dos: mil doscientos veinticinco pares. [[b]]En código es una sola línea: el triángulo superior con np.triu, y después stack.",
    CL(), T("50 × 49 / 2 =", 760, 300, tam=72, c="tinta", en="a"), T("1.225", 1200, 300, tam=90, c="teal", en="a", t=0.6, anc="lm"), T("pares", 1460, 300, tam=48, c="teal", w="sb", en="a", t=1.0, anc="lm"),
    codigo(["corr = X.corr().abs()", "sup = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))", "pares = sup.stack().sort_values(ascending=False)"],
           y=600, tam=34, en="b", dt=0.4))
v.b("¿Qué aparece? [[a]]Once pares sobre cero coma noventa. Varios los fabricamos nosotros: [[b]]valor alto y cuarto cuartil del valor del cliente, uno coma cero. Reclamos y tiene reclamo, uno coma cero. [[c]]Y otros venían en la base: segundos y llamadas, cero coma nueve tres cinco; mensajes y valor del cliente, cero coma nueve dos cinco.",
    CL(), T(f"{D['n_pares90']} pares con |r| > 0.90 · las 50 candidatas", 960, 190, tam=40, c="tinta", ent="subir", en="a"),
    *[[Bc(960, 290 + 66 * i, 1300, 56, "blanco", "linea", 2, 14, en=("b" if i < 6 else "c"), ent="subir", t=0.12 * (i if i < 6 else i - 6)),
       T(f"{a}  ↔  {b_}", 340, 290 + 66 * i, tam=29, c="tinta", w="sb", anc="lm", en=("b" if i < 6 else "c"), t=0.12 * (i if i < 6 else i - 6) + 0.05),
       T(f"{r:.3f}", 1580, 290 + 66 * i, tam=30, c="rojo" if r > 0.99 else "ambar", w="b", anc="rm", en=("b" if i < 6 else "c"), t=0.12 * (i if i < 6 else i - 6) + 0.1)]
      for i, (a, b_, r, _g) in enumerate(PA)],
    T("fabricadas por nosotros", 170, 455, tam=26, c="rojo", w="sb", mw=250, en="b", t=0.9), R(960, 290 + 66 * 8.5, 1330, 140, "ambar", 18, en="c", t=1.0),
    T("ya venían en la base", 170, 290 + 66 * 8.5, tam=26, c="ambar", w="sb", mw=250, en="c", t=1.2))

# ---- 5. Desempate
v.b("Detectar el par no es decidir. [[a]]La clase desempata así: se queda la de mayor información mutua con la fuga. [[b]]Segundos tiene cero coma uno tres uno; llamadas, cero coma cero nueve uno. Se queda segundos. [[c]]Y si la señal es pareja, se queda la más simple de explicar y de mantener en producción.",
    CL(), T("cómo se desempata un par redundante", 960, 190, tam=44, c="tinta", ent="subir"),
    P("1 · mayor información mutua con la fuga", 960, 300, "violeta", 44, en="a"),
    Bc(560, 520, 600, 220, "blanco", "verde", 5, 26, sombra=True, en="b", ent="subir"), T("segundos", 560, 480, tam=50, c="verde", w="m", en="b", t=0.1),
    T(f"MI {MP['segundos']:.3f}", 560, 560, tam=44, c="verde", en="b", t=0.2), K(800, 450, "check", 54, en="b", t=0.6),
    Bc(1360, 520, 600, 220, "blanco", "gris", 5, 26, sombra=True, en="b", ent="subir", t=0.2), T("frec_llamadas", 1360, 480, tam=50, c="gris", w="m", en="b", t=0.3),
    T(f"MI {MP['frec_llamadas']:.3f}", 1360, 560, tam=44, c="gris", en="b", t=0.4),
    P("2 · a igualdad: la más simple de explicar y mantener", 960, 760, "teal", 44, en="c"), sec="5. Cómo se desempata")
v.b("Una precisión que vale la pena. [[a]]Reclamos y tiene reclamo son columnas idénticas, [[b]]y aun así la información mutua les da valores distintos: cero coma cero ocho cuatro y cero coma uno cero dos. [[c]]Es ruido de la estimación. Con dos columnas idénticas, decide el segundo criterio: la más simple, reclamos.",
    CL(), P("reclamos = tiene_reclamo  (r = 1.000)", 960, 260, "rojo", 50, en="a"),
    *[[Bc(x, 500, 520, 220, "blanco", c, 5, 26, sombra=True, en="b", ent="subir", t=t_), T(nm, x, 460, tam=46, c=c, w="m", en="b", t=t_ + 0.1),
       T(f"MI {MP[nm]:.3f}", x, 540, tam=44, c=c, en="b", t=t_ + 0.2)] for x, nm, c, t_ in ((560, "reclamos", "azul", 0.0), (1360, "tiene_reclamo", "azul", 0.3))],
    T("≠", 960, 500, tam=90, c="rojo", en="b", t=0.6), P("ruido de estimación → decide el negocio: reclamos", 960, 760, "verde", 48, en="c"))
v.b("Y otra alerta del material. [[a]]La diapositiva de redundancia se titula «once pares con correlación sobre cero coma noventa», pero en su tabla pone llamadas y contactos, con cero coma setenta y dos. [[b]]Ese par no pasa el umbral. [[c]]Y donde dice cero coma nueve cero siete para segundos y llamadas, es el valor con los datos crudos. Después de limpiar, es cero coma nueve tres cinco.",
    CL(), alerta(),
    T("«11 pares sobre 0.90» · frec_llamadas ↔ n_contactos = 0.728", 960, 340, tam=44, c="gris", w="sb", en="a", id="q728"), U("q728", tachado=1.0, en="b", dur=0.6),
    P(f"0.728 < 0.90: no pasa el umbral (limpio: {D['corr_limpio']['frec_llamadas|n_contactos']:.3f})", 960, 480, "rojo", 42, en="b", t=0.3),
    Bc(560, 700, 600, 200, "blanco", "gris", 4, 26, en="c", ent="subir"), T("datos crudos", 560, 660, tam=36, c="gris", w="sb", en="c", t=0.1),
    N(D["corr_raw"]["segundos|frec_llamadas"], 560, 730, "{:.3f}", 60, "gris", en="c", t=0.2),
    Bc(1360, 700, 600, 200, "blanco", "verde", 4, 26, en="c", ent="subir", t=0.4), T("después de limpiar", 1360, 660, tam=36, c="verde", w="sb", en="c", t=0.5),
    N(D["corr_limpio"]["segundos|frec_llamadas"], 1360, 730, "{:.3f}", 60, "verde", en="c", t=0.6),
    T("segundos ↔ frec_llamadas: las dos cifras son correctas, de datos distintos", 960, 880, tam=32, c="gris", w="sb", en="c", t=1.0))

# ---- 6. Error típico
v.b("El error típico con la redundancia: [[a]]quedarse con la primera de la lista. [[b]]Como dice la clase, ese criterio es el orden alfabético disfrazado de método.",
    CL(), T("«me quedo con la primera de la lista»", 960, 360, tam=64, c="tinta", en="a", id="pri"), U("pri", tachado=1.0, color="gris", en="b", t=0.4, dur=0.6),
    P("= orden alfabético disfrazado de método", 960, 580, "rojo", 56, en="b", t=0.6), sec="6. El error típico")

# ---- 7. Piensa un segundo
v.b("Piensa un segundo. [[a]]Mensajes y valor del cliente tienen correlación cero coma nueve dos cinco. ¿Con cuál te quedas, y por qué?",
    CL(), T("frec_sms  ↔  clv", 960, 330, tam=72, c="violeta", w="m", en="a"), N(0.925, 960, 470, "r = {:.3f}", 90, "rojo", en="a", t=0.4),
    T("¿cuál se queda, y por qué?", 960, 650, tam=60, c="tinta", en="a", t=1.0), sec="7. Piensa un segundo", pausa=0.6)
v.b("Con el valor del cliente. [[a]]Su información mutua con la fuga es cero coma uno cero dos, contra cero coma cero cinco ocho de los mensajes. [[b]]Y si hubieran salido parejas, decidiría cuál es más fácil de explicar y de mantener.",
    CL(), P("se queda clv", 960, 240, "verde", 64, ent="subir"),
    Bx(560, 760 - MP["clv"] * 3000, 820, 760, "verde", None, 0, 10, grow="y", en="a"), N(MP["clv"], 690, 760 - MP["clv"] * 3000 - 40, "{:.3f}", 56, "verde", en="a", t=0.3),
    T("clv", 690, 800, tam=36, c="verde", w="m", en="a"),
    Bx(1100, 760 - MP["frec_sms"] * 3000, 1360, 760, "gris", None, 0, 10, grow="y", en="a", t=0.3), N(MP["frec_sms"], 1230, 760 - MP["frec_sms"] * 3000 - 40, "{:.3f}", 56, "gris", en="a", t=0.6),
    T("frec_sms", 1230, 800, tam=36, c="gris", w="m", en="a", t=0.3), L(460, 760, 1460, 760, "tinta", 4),
    T("información mutua con la fuga", 960, 870, tam=32, c="gris", w="sb", en="a", t=0.8), P("si empatan: la más simple de mantener", 1500, 420, "teal", 38, en="b"))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]irrelevante se descarta, redundante deja una, y la útil en conjunto no se mata por un p-valor. [[b]]Número de fallas lo demostró: sola, azar; acompañada, sube el modelo. [[c]]Y un par redundante se desempata por información mutua y por simplicidad, nunca por la lista.",
    CL(), P("irrelevante → fuera · redundante → una · en conjunto → se queda", 960, 300, "violeta", 44, en="a"),
    P(f"numero_fallas: sola {FA['sola']:.3f} · acompañada sube a {FA['base_f']:.3f}", 960, 480, "verde", 46, en="b"),
    P("desempate: información mutua, después simplicidad", 960, 660, "teal", 46, en="c"), sec="8. Para llevar")
v.b("Siguiente parada: [[a]]el caso extremo de la variable útil solo en conjunto. El XOR.",
    CL(), siguiente(5, "El caso XOR: útil solo en conjunto"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
