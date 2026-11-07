import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, caja_formula, alerta

OUT = out("guion_c33_08_wrapper.json")
v = V("Wrapper y embedded: RFE y Lasso")
SC, LC, LT, RS, ES = D["subconj"], D["lasso_C"], D["lasso_top"], D["resumen"], D["estab"]

v.portada("RFE y Lasso", "Los métodos que sí miran el conjunto, y cuánto cuestan",
          "Séptima parada: wrapper y embedded. Los métodos que sí miran el conjunto de variables.")

# ---- 1. Hook: combinaciones de ropa
v.b("Imagina que tienes diez prendas y quieres el mejor conjunto para una entrevista. [[a]]Cada prenda entra o no entra: dos posibilidades por prenda. [[b]]Dos elevado a diez son mil veinticuatro. Menos el conjunto vacío: mil veintitrés combinaciones.",
    CL(), P("10 prendas · ¿cuál es el mejor conjunto?", 960, 210, "violeta", 52, ent="subir"),
    *[[Bc(330 + 140 * i, 400, 110, 110, "blanco", "violeta", 4, 18, t=0.08 * i, ent="pop"), T("sí/no", 330 + 140 * i, 400, tam=26, c="violeta", w="sb", en="a", t=0.08 * i)] for i in range(10)],
    T("2 · 2 · 2 · … · 2 = 2¹⁰ =", 860, 600, tam=56, c="tinta", en="b"), T("1.024", 1320, 600, tam=72, c="violeta", en="b", t=0.5, anc="lm"),
    T("− 1 (vacío) =", 860, 720, tam=56, c="tinta", en="b", t=1.0), T("1.023", 1320, 720, tam=72, c="ambar", en="b", t=1.4, anc="lm"),
    ilustrativo(1000, texto="analogía"), sec="1. Probar todas las combinaciones")
v.b("Un wrapper hace eso con variables: entrena el modelo con distintos subconjuntos y compara. [[a]]Con p variables hay dos elevado a p, menos uno, subconjuntos posibles. [[b]]Y el número explota.",
    CL(), caja_formula("subconjuntos = 2^p − 1", y=330, c="ambar", tam=66, w=1000, en="a"),
    P("crece exponencialmente", 960, 560, "rojo", 56, en="b"))

# ---- 2. Ejemplo: el costo a 1.000 modelos por segundo
FIL = [("10", "1.023", "1 segundo", "verde"), ("20", "1.048.575", "17 minutos", "teal"), ("30", "≈ 1.07 · 10⁹", "12 días", "ambar"),
       ("50", "≈ 1.13 · 10¹⁵", "35.702 años", "rojo")]
v.b("Supongamos un computador que entrena mil modelos por segundo. [[a]]Con diez variables: mil veintitrés modelos, un segundo. [[b]]Con veinte: un millón cuarenta y ocho mil, diecisiete minutos. [[c]]Con treinta: mil millones, doce días. [[d]]Y con nuestras cincuenta candidatas: treinta y cinco mil setecientos dos años.",
    CL(), T("¿cuánto cuesta probar todo? (1.000 modelos por segundo)", 960, 190, tam=40, c="tinta", ent="subir"),
    T("p", 360, 290, tam=34, c="gris", w="sb"), T("subconjuntos", 900, 290, tam=34, c="gris", w="sb"), T("tiempo", 1440, 290, tam=34, c="gris", w="sb"),
    *[[Bc(960, 390 + 130 * i, 1400, 104, "blanco", c, 4, 22, en=k, ent="subir"), T(p_, 360, 390 + 130 * i, tam=52, c=c, en=k, t=0.1),
       T(s_, 900, 390 + 130 * i, tam=48, c="tinta", w="sb", en=k, t=0.2), T(t_, 1440, 390 + 130 * i, tam=50, c=c, en=k, t=0.4)]
      for i, ((p_, s_, t_, c), k) in enumerate(zip(FIL, "abcd"))],
    T(f"verificado: (2⁵⁰ − 1) / 1.000 s = {D['anios50']:,} años".replace(",", "."), 960, 930, tam=32, c="gris", w="sb", en="d", t=0.8),
    sec="2. Ejemplo: el costo de probar todo")
v.b("Por eso nadie hace búsqueda exhaustiva. [[a]]Todos los métodos prácticos son atajos. [[b]]El más conocido es RFE: eliminación recursiva de variables.",
    CL(), T("búsqueda exhaustiva", 960, 330, tam=70, c="gris", id="bx"), U("bx", tachado=1.0, en="a", dur=0.5),
    P("los métodos prácticos son atajos", 960, 500, "teal", 54, en="a", t=0.4), P("RFE · Recursive Feature Elimination", 960, 680, "azul", 54, en="b"))

# ---- 3. RFE
PASOS = ["entrena con las p variables", "ordena por |β| y encuentra la más débil", "la elimina y reentrena con p − 1", "repite hasta el mínimo pedido"]
v.b("RFE es literalmente recursivo. [[a]]Uno: entrena el modelo con todas las variables. [[b]]Dos: ordena las variables por el tamaño de su coeficiente, y encuentra la más débil. [[c]]Tres: la elimina, y vuelve a entrenar. [[d]]Cuatro: repite, hasta llegar al mínimo pedido.",
    CL(), T("RFE · eliminación recursiva", 960, 190, tam=48, c="azul", ent="subir"),
    *[[Bc(960, 320 + 130 * i, 1100, 100, "blanco", "azul", 4, 22, en=k, ent="subir"), T(str(i + 1), 470, 320 + 130 * i, tam=56, c="azul", en=k, t=0.1),
       T(tx, 1000, 320 + 130 * i, tam=40, c="tinta", w="sb", en=k, t=0.2)] for i, (tx, k) in enumerate(zip(PASOS, "abcd"))],
    L(1560, 320 + 130 * 3, 1560, 320 + 30, "gris", 5, True, en="d", t=0.5), sec="3. Wrapper: RFE")
v.b("La versión con validación cruzada, RFECV, hace todo ese proceso dentro de cada fold, [[a]]y elige el número de variables que maximiza el AUC promedio. [[b]]Con las cincuenta candidatas de la clase, RFECV se queda con cuarenta y dos.",
    CL(), T("RFECV(estimador, step=1, cv=5, scoring='roc_auc')", 960, 240, tam=40, c="tinta", w="m", ent="subir"),
    P("elige el k que maximiza el AUC promedio de los folds", 960, 400, "azul", 46, en="a"),
    Bc(960, 640, 640, 300, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), N(D["rfe_n"], 960, 610, "{:.0f}", 130, "azul", en="b", t=0.1),
    T("de 50 variables · base real", 960, 730, tam=36, c="azul", w="sb", en="b", t=0.3))
v.b("Un detalle clave: hay que escalar antes. [[a]]RFE ordena por el tamaño de los coeficientes. El valor del cliente llega a dos mil ciento sesenta y cinco; tiene reclamo vale cero o uno. [[b]]Sin escalar, el coeficiente del valor del cliente sale diminuto por la unidad de medida, no por falta de importancia. [[c]]Por eso el StandardScaler va dentro del pipeline.",
    CL(), P("escalar antes de RFE o Lasso", 960, 210, "ambar", 54, ent="subir"),
    tarjetas([(560, "violeta", "clv", "de 0 a 2.165 → coeficiente diminuto", "a"), (1360, "teal", "tiene_reclamo", "0 o 1 → coeficiente grande", "a")], y=470, w=640, h=280),
    P("el tamaño refleja la unidad, no la importancia", 960, 700, "rojo", 46, en="b"),
    T("Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])", 960, 840, tam=36, c="tinta", w="m", en="c"))

# ---- 4. Lasso
BETAS = [0.9, -0.7, 0.5, 0.3, -0.2, 0.12]
v.b("Ahora, embedded: la selección ocurre mientras el modelo se entrena. [[a]]El ejemplo clásico es el Lasso: una regresión logística con penalización ele uno, la suma de los valores absolutos de los coeficientes. [[b]]Esa penalización empuja coeficientes a exactamente cero. Y una variable con coeficiente cero queda fuera.",
    CL(), T("Embedded: Lasso (penalización L1)", 960, 190, tam=48, c="verde", ent="subir"),
    caja_formula("penalización = Σ |βⱼ|", y=330, c="verde", tam=54, w=900, en="a"),
    L(300, 690, 1620, 690, "tinta", 3),
    *[Bx(380 + 220 * i - 70, 690 - max(b_, 0) * 260, 380 + 220 * i + 70, 690 - min(b_, 0) * 260, "verde" if b_ > 0 else "azul", None, 0, 8, id=f"lb{i}", t=0.1 * i, ent="fade") for i, b_ in enumerate(BETAS)],
    *[U(f"lb{i}", y0=690 - max(b_ * (0.55 if i < 3 else 0), 0) * 260, y1=690 - min(b_ * (0.55 if i < 3 else 0), 0) * 260, en="b", t=0.2, dur=1.4) for i, b_ in enumerate(BETAS)],
    P("tres coeficientes llegan a 0 → fuera", 1300, 880, "verde", 40, en="b", t=1.6), ilustrativo(1000), sec="4. Embedded: el Lasso")
v.b("¿Y Ridge, la penalización ele dos? [[a]]Achica los coeficientes, pero casi nunca los lleva a cero. [[b]]Por eso Ridge no sirve para seleccionar. El Lasso sí.",
    CL(), tarjetas([(560, "azul", "Ridge · L2", "achica, casi nunca llega a 0 → no selecciona", "a"), (1360, "verde", "Lasso · L1", "ceros exactos → selecciona", "b")], y=480, w=700, h=340))
v.b("Y una trampa de sklearn: [[a]]C no es la fuerza de la penalización. Es su inverso: en la fórmula, C multiplica al error, no a la penalización. [[b]]C chico: el error pesa poco, la penalización manda, y sobreviven menos variables. [[c]]Y ojo: el Lasso necesita solver liblinear o saga. El solver por defecto no admite ele uno.",
    CL(), caja_formula("min  Σ|β|  +  C · error", y=260, c="violeta", tam=60, w=1000, en="a"),
    P("C chico → penalización fuerte → menos variables", 960, 460, "rojo", 48, en="b"),
    T("LogisticRegression(penalty='l1', solver='liblinear', C=0.05)", 960, 640, tam=38, c="tinta", w="m", en="c"),
    T("el solver por defecto (lbfgs) no admite L1", 960, 740, tam=36, c="gris", w="sb", en="c", t=0.6))

# ---- 5. C real
CS_ = [0.01, 0.05, 0.1, 0.5, 1.0]
NV = [LC[str(c)] for c in CS_]
GC = dict(x0=300, y0=280, x1=1200, y1=820)
v.b("Con los datos de la clase. [[a]]C igual a cero coma cero uno deja trece variables vivas. [[b]]Cero coma cero cinco, veintisiete. Cero coma uno, treinta y tres. [[c]]Cero coma cinco, cuarenta y dos. Y C igual a uno, cuarenta y cinco.",
    CL(), T("variables vivas según C · Lasso sobre las 50 candidatas", 960, 190, tam=40, c="tinta", ent="subir"),
    G(GC["x0"], GC["y0"], GC["x1"], GC["y1"], -2.1, 0.1, 0, 50, [dict(xs=[math.log10(c) for c in CS_], ys=NV, color="violeta", grosor=6, marcadores=11, revela=0.0)],
      xticks=[], yticks=[0, 10, 20, 30, 40, 50], yfmt="{:.0f}", xlabel="C (escala logarítmica)", id="gc"),
    *[T(f"{c:g}", GC["x0"] + (math.log10(c) + 2.1) / 2.2 * (GC["x1"] - GC["x0"]), GC["y1"] + 22, tam=28, c="gris") for c in CS_],
    U("gc", **{"series.0.revela": 0.02}, en="a", dur=0.3), U("gc", **{"series.0.revela": 0.62}, en="b", dur=1.2), U("gc", **{"series.0.revela": 1.0}, en="c", dur=1.0),
    *[T(f"C = {c:g} → {n_}", 1520, 330 + 95 * i, tam=42, c="violeta", w="sb", en=k, t=t_) for i, (c, n_, k, t_) in enumerate(zip(CS_, NV, "abbcc", (0.3, 0.0, 0.9, 0.0, 1.0)))],
    sec="5. El efecto de C")
v.b("Con C igual a cero coma cero cinco, el valor de la clase, sobreviven veintisiete de cincuenta. [[a]]Los coeficientes más grandes: llamadas, monto, cliente nuevo, [[b]]y en cuarto lugar, número de fallas, con más cero coma cinco uno siete. La variable que el filtro llamaba ruido. [[c]]Más abajo, reclamos y tiene reclamo sobreviven las dos, aunque son idénticas.",
    CL(), T(f"Lasso C = 0.05 · {D['lasso_n']} de 50 sobreviven · coeficientes estandarizados", 960, 180, tam=36, c="tinta", ent="subir"),
    L(960, 240, 960, 900, "tinta", 3),
    *[[T(nm, 940 if b_ > 0 else 980, 280 + 80 * i, tam=32, c="tinta", w="m", anc="rm" if b_ > 0 else "lm", en=k, t=t_),
       Bx(960 if b_ > 0 else 960 + b_ * 700, 280 + 80 * i - 26, 960 + b_ * 700 if b_ > 0 else 960, 280 + 80 * i + 26, "verde" if b_ > 0 else "azul", None, 0, 6, grow="x", en=k, t=t_ + 0.05),
       T(f"{b_:+.3f}".replace("-", "−"), (960 + b_ * 700 + 14) if b_ > 0 else (960 + b_ * 700 - 14), 280 + 80 * i, tam=30, c="verde" if b_ > 0 else "azul", w="sb",
         anc="lm" if b_ > 0 else "rm", en=k, t=t_ + 0.3)]
      for i, ((nm, b_), k, t_) in enumerate(zip(LT, ["a", "a", "a", "b", "c", "c", "c", "c"], [0, 0.3, 0.6, 0, 0, 0.3, 0.6, 0.9]))],
    R(960, 280 + 80 * 3, 1500, 70, "ambar", 18, en="b", t=0.5), sec="6. Lo que deja el Lasso")
v.b("Y aquí, una alerta del material. [[a]]La diapositiva veinticinco dice «veinticinco de cincuenta sobreviven», y la veintiséis, cero coma nueve tres ocho con veinticinco variables. [[b]]El notebook ejecutado da veintisiete de cincuenta, con AUC cero coma nueve tres nueve seis. Lo reproduje con la base: el valor correcto es el del notebook.",
    CL(), alerta(),
    T("«25 de 50 sobreviven» · «0.938 con 25»", 960, 360, tam=58, c="gris", en="a", id="q25"), U("q25", tachado=1.0, en="b", dur=0.6),
    P(f"reproducido: {D['lasso_n']} de 50 · AUC {RS[2][2]:.4f}", 960, 560, "verde", 56, en="b", t=0.4), K(1560, 560, "check", 56, en="b", t=0.9))

# ---- 7. Resumen de estrategias
v.b("¿Qué logramos con cada estrategia? AUC con validación cruzada. [[a]]Las doce originales, cero coma nueve dos. Las cincuenta candidatas, cero coma nueve cuatro uno. [[b]]El Lasso, con veintisiete, cero coma nueve tres nueve seis: casi lo mismo, con la mitad de las variables. [[c]]El top cuatro, cero coma ocho cero cuatro. Y cuatro sin redundancia, cero coma ocho nueve tres.",
    CL(), T("resumen de estrategias · AUC con validación cruzada de 5 folds", 960, 180, tam=38, c="tinta", ent="subir"),
    *[[T(f"{nm}", 600, 300 + 120 * i, tam=38, c=c, w="sb", anc="rm", en=k, t=t_),
       Bx(630, 300 + 120 * i - 34, 630 + (a - 0.75) * 4800, 300 + 120 * i + 34, c, None, 0, 8, grow="x", en=k, t=t_ + 0.05),
       N(a, 630 + (a - 0.75) * 4800 + 20, 300 + 120 * i, "{:.4f}", 40, c, en=k, t=t_ + 0.3, anc="lm")]
      for i, ((nm, _n, a), c, k, t_) in enumerate(zip(RS, ["gris", "violeta", "verde", "rojo", "teal"], ["a", "a", "b", "c", "c"], [0, 0.5, 0, 0, 0.5]))],
    T("eje desde 0.75", 1700, 900, tam=26, c="gris", w="sb"), sec="7. Resumen de estrategias")

# ---- 8. Estabilidad
v.b("Pero hay una pregunta incómoda: [[a]]si repites el Lasso con otra partición de los datos, ¿salen las mismas variables? La clase lo probó con cinco particiones distintas. [[b]]Dieciséis variables aparecen en las cinco. Diecinueve aparecen solo en algunas. Y quince, en ninguna.",
    CL(), T("estabilidad del Lasso · 5 particiones (75 % de los datos cada una)", 960, 190, tam=38, c="tinta", ent="subir"),
    Pt([(ES["siempre"], [260, 380, 760, 640], "verde"), (ES["a_veces"], [760 + 60, 380, 1320, 640], "ambar"), (ES["nunca"], [1380, 380, 1660, 640], "#CBD5E1")], 20,
       visibles=50, en="a", dur=1.6),
    N(ES["siempre"], 510, 720, "{:.0f}", 80, "verde", en="b"), T("en las 5", 510, 800, tam=34, c="verde", w="sb", en="b"),
    N(ES["a_veces"], 1070, 720, "{:.0f}", 80, "ambar", en="b", t=0.4), T("solo en algunas", 1070, 800, tam=34, c="ambar", w="sb", en="b", t=0.4),
    N(ES["nunca"], 1520, 720, "{:.0f}", 80, "gris", en="b", t=0.8), T("en ninguna", 1520, 800, tam=34, c="gris", w="sb", en="b", t=0.8), sec="8. ¿Es estable?")
v.b("Las inestables son casi todas variables construidas, y correlacionadas entre sí. [[a]]Cuando dos variables llevan la misma información, el Lasso elige una casi al azar. [[b]]Una variable que aparece en cinco de cinco es defendible ante un comité. Una que aparece en dos de cinco, no.",
    CL(), P("inestables = construidas y correlacionadas", 960, 260, "ambar", 52, ent="subir"),
    T("dos variables con la misma información → el Lasso elige una casi al azar", 960, 430, tam=44, c="tinta", w="sb", mw=1600, en="a"),
    tarjetas([(560, "verde", "5 de 5", "defendible ante un comité", "b"), (1360, "rojo", "2 de 5", "el próximo reentrenamiento la cambia", "b")], y=690, w=640, h=280))

# ---- 9. Error típico
v.b("El error típico con el Lasso: [[a]]leer C como si fuera la fuerza de la penalización, y subirlo para quitar variables. [[b]]Es al revés: subir C deja más variables vivas.",
    CL(), T("«subo C para penalizar más»", 960, 360, tam=70, c="tinta", en="a", id="sc"), U("sc", tachado=1.0, color="gris", en="b", t=0.1, dur=0.6),
    P("subir C → menos penalización → más variables", 960, 580, "rojo", 54, en="b", t=0.5), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con C igual a cero coma cero cinco quedan veintisiete variables. Si cambias a C igual a cero coma cinco, ¿quedan más o menos?",
    CL(), T("C = 0.05 → 27 variables", 960, 330, tam=70, c="violeta", en="a"), T("C = 0.5 → ¿más o menos?", 960, 520, tam=70, c="tinta", en="a", t=1.0),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("Más: cuarenta y dos. [[a]]C subió diez veces, así que la penalización bajó, y sobrevivieron quince variables más.",
    CL(), N(LC["0.5"], 960, 360, "{:.0f}", 160, "verde"), T("variables con C = 0.5", 960, 500, tam=44, c="verde", w="sb", t=0.3),
    P("C × 10 → penalización relativa ÷ 10 → +15 variables", 960, 680, "teal", 52, en="a"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el wrapper prueba subconjuntos; la búsqueda exhaustiva es imposible, y RFE es el atajo. [[b]]El Lasso selecciona al entrenar, y en sklearn, C chico deja menos variables. Siempre con variables escaladas. [[c]]Y mide la estabilidad: solo dieciséis variables aparecen en las cinco particiones.",
    CL(), P("wrapper: RFE · exhaustiva imposible (35.702 años)", 960, 300, "azul", 48, en="a"),
    P("Lasso: C chico → menos variables · escalar antes", 960, 480, "verde", 48, en="b"), P(f"estabilidad: {ES['siempre']} aparecen en 5 de 5", 960, 660, "ambar", 48, en="c"),
    sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la trampa. Todo lo que hicimos hasta aquí tiene un problema, y lo vamos a medir.",
    CL(), siguiente(9, "La trampa: dónde vive la selección"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
