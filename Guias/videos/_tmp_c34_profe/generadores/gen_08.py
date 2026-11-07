import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, codigo

OUT = out("guion_c34_08_importancia.json")
v = V("Importancia: impureza contra permutación")
IM = D["imp"]
TOP = IM[:10]
RK_I = {r["var"]: i + 1 for i, r in enumerate(IM)}
RK_P = {r["var"]: i + 1 for i, r in enumerate(sorted(IM, key=lambda r: -r["perm"]))}
I = {r["var"]: r for r in IM}
clv, est, fal, ant = I["clv"], I["estado"], I["numero_fallas"], I["antiguedad"]

v.portada("Importancia de variables", "Por impureza o por permutación: una de las dos engaña",
          "Octava parada. ¿Qué variables importan en un modelo de árboles? Hay dos maneras de medirlo, y dan respuestas distintas. Hoy aprendes cuál creerle.")

# ---- 1. Hook
v.b("El árbol de profundidad tres usó seis variables de doce: eso ya es selección. [[a]]Pero para un bosque de trescientos árboles necesitas un número por variable. [[b]]sklearn te lo da en una línea: feature_importances_. [[c]]El problema es que ese número engaña, de una forma muy precisa.",
    CL(), T("¿qué variables importan?", 960, 230, tam=60, c="tinta", ent="subir"),
    P("un bosque de 300 árboles: un número por variable", 960, 400, "azul", 46, en="a"),
    T("modelo.feature_importances_", 960, 560, tam=56, c="tinta", w="m", en="b"), P("engaña", 960, 720, "rojo", 64, en="c"), sec="1. Dos formas de medir")

# ---- 2. Importancia por impureza
v.b("Primero, la importancia por impureza. [[a]]Suma, para cada variable, la ganancia de todos los cortes donde participó, ponderada por cuántos clientes pasaban por cada nodo. [[b]]Y se normaliza para que todas sumen uno.",
    CL(), T("importancia por impureza", 960, 190, tam=52, c="violeta", ent="subir"),
    caja_formula("imp(Xⱼ) = Σ nodos con Xⱼ  (n_nodo / n) · ganancia", y=400, c="violeta", tam=48, en="a", w=1450),
    P("normalizada: todas suman 1", 960, 600, "teal", 48, en="b"), sec="2. Importancia por impureza")
v.b("Su sesgo se llama cardinalidad. [[a]]Una variable con muchos valores distintos ofrece muchos umbrales para probar. [[b]]Es como una lotería: quien compra dos mil seiscientos boletos gana más seguido que quien compra dos, aunque no sea más hábil. [[c]]Por puro azar, alguno de tantos cortes reduce algo la impureza, y la importancia por impureza lo premia.",
    CL(), T("el sesgo por cardinalidad", 960, 190, tam=52, c="rojo", ent="subir"),
    Bc(560, 470, 640, 300, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("clv", 560, 380, tam=48, c="violeta", w="m", en="a", t=0.1),
    T(f"{mil(clv['u'])} valores distintos", 560, 470, tam=40, c="tinta", w="sb", en="a", t=0.2), T(f"{mil(clv['u']) } boletos de lotería", 560, 545, tam=32, c="gris", w="sb", en="b"),
    Bc(1360, 470, 640, 300, "blanco", "teal", 5, 30, sombra=True, en="a", ent="subir", t=0.3), T("estado", 1360, 380, tam=48, c="teal", w="m", en="a", t=0.4),
    T(f"{est['u']} valores distintos", 1360, 470, tam=40, c="tinta", w="sb", en="a", t=0.5), T(f"{est['u']} boletos", 1360, 545, tam=32, c="gris", w="sb", en="b", t=0.3),
    P("muchos cortes → alguno acierta por azar", 960, 760, "rojo", 48, en="c"), ilustrativo(1000, texto="analogía (la lotería)"))

# ---- 3. Importancia por permutación
v.b("La alternativa es la importancia por permutación. [[a]]Tomas datos que el modelo no vio. [[b]]Desordenas al azar una sola columna, como si esa variable se volviera ruido. [[c]]Y mides cuánto cae el AUC. Si el modelo necesitaba esa variable, el AUC cae mucho. Si no, casi nada.",
    CL(), T("importancia por permutación", 960, 180, tam=52, c="verde", ent="subir"),
    *[[Bc(330 + 630 * i, 470, 520, 260, "blanco", c, 5, 30, sombra=True, en=m, ent="subir"), T(t1, 330 + 630 * i, 410, tam=42, c=c, en=m, t=0.1),
       T(t2, 330 + 630 * i, 500, tam=32, c="tinta", w="sb", mw=440, en=m, t=0.2)] + ([L(330 + 630 * i - 360, 470, 330 + 630 * i - 270, 470, "gris", 6, True, en=m)] if i else [])
      for i, (t1, t2, c, m) in enumerate((("1. datos no vistos", "el test: 788 clientes", "azul", "a"), ("2. desordenar Xⱼ", "solo esa columna, al azar", "ambar", "b"),
                                           ("3. medir la caída", "AUC original − AUC desordenado", "verde", "c")))],
    caja_formula("imp_perm(Xⱼ) = AUC − AUC(con Xⱼ desordenada)", y=780, c="verde", tam=44, en="c", t=1.0, w=1300), sec="3. Importancia por permutación")
v.b("Un ejemplo del apunte. [[a]]El modelo tiene AUC cero coma nueve. Desordenas reclamos, y el AUC baja a cero coma ocho seis cinco: importancia cero coma cero tres cinco. [[b]]Desordenas una variable ruidosa, y baja a cero coma ocho nueve nueve: importancia cero coma cero cero uno. [[c]]El modelo casi no la usaba.",
    CL(), P("ejemplo del apunte", 960, 170, "gris", 34, ent="subir"),
    T("reclamos:  0.900 − 0.865 =", 900, 340, tam=52, c="tinta", w="sb", anc="rm", en="a"), N(0.900 - 0.865, 930, 340, "{:.3f}", 76, "rojo", anc="lm", en="a", t=0.6),
    T("variable ruidosa:  0.900 − 0.899 =", 900, 500, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(0.900 - 0.899, 930, 500, "{:.3f}", 76, "gris", anc="lm", en="b", t=0.6),
    P("si el modelo no la necesita, el AUC casi no cae", 960, 690, "verde", 46, en="c"))
v.b("Una regla sin excepciones: [[a]]la permutación se mide sobre datos que el modelo no vio. [[b]]Sobre el entrenamiento, hereda el sobreajuste: mide memoria, no capacidad de predecir. [[c]]Es más lenta, porque repite el desorden veinte veces, pero mide lo que uno cree que está midiendo.",
    CL(), P("SIEMPRE sobre datos no vistos", 960, 280, "verde", 60, en="a"),
    T("sobre el entrenamiento: mide memoria", 960, 440, tam=48, c="rojo", w="sb", en="b"),
    T("permutation_importance(rf, X_test, y_test, n_repeats=20, scoring='roc_auc')", 960, 620, tam=34, c="tinta", w="m", en="c"),
    T("más lenta · mide lo que crees que mide", 960, 720, tam=38, c="gris", w="sb", en="c", t=0.5))

# ---- 4. Con los datos de la clase
XI, XP, Y0, DY = 640, 1330, 280, 62
v.b("Ahora con los datos de la clase: un Random Forest de trescientos árboles, entrenado con el setenta y cinco por ciento, y medido en setecientos ochenta y ocho clientes de prueba. [[a]]A la izquierda, la importancia por impureza. [[b]]A la derecha, por permutación.",
    CL(), T("Random Forest (300 árboles) · test de 788 clientes", 960, 160, tam=40, c="tinta", ent="subir"),
    T("por impureza", XI + 200, 220, tam=34, c="violeta", w="b", en="a"), T("por permutación", XP + 200, 220, tam=34, c="verde", w="b", en="b"),
    *[[T(r["var"], XI - 20, Y0 + DY * i, tam=30, c="tinta", w="sb", anc="rm", id=f"nv{i}"),
       Bx(XI, Y0 + DY * i - 18, XI + r["imp"] * 2400, Y0 + DY * i + 18, "violeta", None, 0, 6, grow="x", en="a", t=0.06 * i, id=f"bi{i}"),
       T(f"{r['imp']:.3f}", XI + r["imp"] * 2400 + 12, Y0 + DY * i, tam=26, c="violeta", w="b", anc="lm", en="a", t=0.06 * i + 0.3),
       Bx(XP, Y0 + DY * i - 18, XP + max(r["perm"] * 14000, 3), Y0 + DY * i + 18, "verde", None, 0, 6, grow="x", en="b", t=0.06 * i, id=f"bp{i}"),
       T(f"{r['perm']:.4f}", XP + max(r["perm"] * 14000, 3) + 12, Y0 + DY * i, tam=26, c="verde", w="b", anc="lm", en="b", t=0.06 * i + 0.3)]
      for i, r in enumerate(TOP)],
    real(1050), sec="4. Con los datos de la clase")
iC, iE, iF = [next(i for i, r in enumerate(TOP) if r["var"] == k) for k in ("clv", "estado", "numero_fallas")]
v.b("Mira tres variables. [[a]]Ce ele ve: quinta por impureza, con cero coma uno cero nueve. Por permutación cae al séptimo lugar, con cero coma cero uno cuatro. [[b]]Estado, que es binaria: décima por impureza, y sexta por permutación, por encima de ce ele ve. [[c]]Y número de fallas sube del octavo al quinto lugar.",
    R(XI + 150, Y0 + DY * iC, 1500, 52, "rojo", 16, en="a", pulso=False), R(XI + 150, Y0 + DY * iE, 1500, 52, "teal", 16, en="b", pulso=False),
    R(XI + 150, Y0 + DY * iF, 1500, 52, "ambar", 16, en="c", pulso=False),
    T(f"clv: {RK_I['clv']}.º → {RK_P['clv']}.º", 120, 900, tam=34, c="rojo", w="b", anc="lm", en="a"),
    T(f"estado: {RK_I['estado']}.º → {RK_P['estado']}.º", 120, 950, tam=34, c="teal", w="b", anc="lm", en="b"),
    T(f"fallas: {RK_I['numero_fallas']}.º → {RK_P['numero_fallas']}.º", 120, 1000, tam=34, c="ambar", w="b", anc="lm", en="c"))

# ---- 5. Ejemplo: el ranking se reordena
v.b("Hagamos la cuenta que delata el sesgo. [[a]]Por impureza, ce ele ve vale cero coma uno cero nueve, y estado cero coma cero cuatro cinco: ce ele ve parece dos coma cuatro veces más importante. [[b]]Por permutación, ce ele ve vale cero coma cero uno cuatro, y estado cero coma cero uno ocho cinco: [[c]]ahora estado vale uno coma tres veces lo que ce ele ve. El orden se dio vuelta.",
    CL(), T("clv contra estado", 960, 170, tam=52, c="tinta", ent="subir"),
    T(f"impureza:  {clv['imp']:.3f} / {est['imp']:.3f} =", 900, 340, tam=50, c="violeta", w="sb", anc="rm", en="a"),
    N(clv["imp"] / est["imp"], 930, 340, "{:.1f}", 76, "violeta", suf=" ×  (gana clv)", anc="lm", en="a", t=0.6),
    T(f"permutación:  {est['perm']:.4f} / {clv['perm']:.4f} =", 900, 500, tam=50, c="verde", w="sb", anc="rm", en="b"),
    N(est["perm"] / clv["perm"], 930, 500, "{:.1f}", 76, "verde", suf=" ×  (gana estado)", anc="lm", en="c"),
    P(f"clv: {mil(clv['u'])} valores · estado: {est['u']} valores", 960, 690, "ambar", 46, en="c", t=0.6),
    P("la impureza premió la cardinalidad, no el poder predictivo", 960, 820, "rojo", 44, en="c", t=1.2), real(1050), sec="5. Ejemplo: el orden se da vuelta")

# ---- 6. Alerta del material
v.b("Una alerta pequeña del material. [[a]]El notebook de la clase dice que ce ele ve tiene la cuarta importancia por impureza más alta. [[b]]Es la quinta: antigüedad está por encima, con cero coma uno uno tres. [[c]]Pasa porque la tabla de la diapositiva veintiuno no muestra antigüedad. La conclusión no cambia.",
    CL(), alerta(),
    T("notebook: clv tiene «la cuarta más alta» por impureza", 960, 330, tam=44, c="tinta", w="sb", en="a", id="nb"),
    U("nb", tachado=1.0, color="gris", en="b", dur=0.6),
    P(f"es la quinta: antigüedad {ant['imp']:.3f} > clv {clv['imp']:.3f}", 960, 470, "verde", 48, en="b", t=0.4),
    T("la diapositiva 21 omite antigüedad (muestra 7 de 12)", 960, 600, tam=38, c="gris", w="sb", en="c"), sec="6. Alerta del material")

# ---- 7. Error típico
v.b("El error típico: [[a]]reportar feature_importances_ como «lo que más importa». [[b]]Tiene dos problemas: premia la cardinalidad, y además no dice la dirección del efecto. Una importancia alta no dice si la variable sube o baja la fuga. [[c]]Usa permutación sobre el test, y para la dirección, lee las reglas del árbol.",
    CL(), T("reportar feature_importances_ tal cual", 960, 260, tam=56, c="tinta", en="a", id="e1"),
    tarjetas([(560, "rojo", "sesgo", "premia la cardinalidad", "b"), (1360, "rojo", "sin dirección", "no dice si sube o baja la fuga", "b")], y=500, w=640, h=260),
    U("e1", tachado=1.0, color="gris", en="c", dur=0.6), P("permutación sobre el test · reglas para la dirección", 960, 790, "verde", 46, en="c", t=0.4),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Agregas al modelo una columna con el número de identificación de cada cliente: tres mil ciento cincuenta valores distintos, sin ninguna relación con la fuga. [[b]]¿Qué le pasa en cada importancia?",
    CL(), P("id_cliente: 3.150 valores distintos · puro ruido", 960, 320, "violeta", 48, en="a"),
    T("¿importancia por impureza? ¿por permutación?", 960, 520, tam=56, c="tinta", en="b"), ilustrativo(1000, texto="ejemplo hipotético"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Por impureza puede salir sorprendentemente alta: tiene miles de cortes para probar, y alguno acierta por azar. [[a]]Por permutación, cerca de cero: desordenarla en el test no cambia nada, porque nunca predijo. [[b]]Es la prueba más clara del sesgo por cardinalidad.",
    CL(), tarjetas([(560, "violeta", "impureza", "alta: miles de cortes, alguno acierta", None), (1360, "verde", "permutación", "≈ 0: nunca predijo", "a")], y=430, w=660, h=300),
    P("la prueba más clara del sesgo por cardinalidad", 960, 720, "rojo", 48, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]la importancia por impureza premia a las variables con muchos valores. [[b]]La importancia por permutación mide cuánto cae el AUC si la variable se vuelve ruido, [[c]]y se calcula siempre sobre datos no vistos.",
    CL(), P("impureza: premia la cardinalidad", 960, 300, "rojo", 54, en="a"),
    P("permutación: cuánto cae el AUC", 960, 480, "verde", 54, en="b"), P("siempre sobre datos no vistos", 960, 660, "teal", 54, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]las fortalezas y los límites del árbol. Lo que no le importa, como escalar, y lo que sí le duele, la inestabilidad.",
    CL(), siguiente(9, "Fortalezas y límites del árbol"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
