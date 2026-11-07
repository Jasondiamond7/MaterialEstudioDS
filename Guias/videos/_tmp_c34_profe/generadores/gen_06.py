import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula

OUT = out("guion_c34_06_sobreajuste.json")
v = V("Sobreajuste y validación cruzada")
RO, CU = D["rondas"], D["curva"]
ETQ = [str(c["d"]) if c["d"] else "sin límite" for c in CU]
best = max(CU, key=lambda c: c["va"])
ib = CU.index(best)
SL = CU[-1]
c3, c5, c6 = CU[2], CU[4], CU[5]
brecha = lambda c: (c["tr"] - c["va"]) * 100  # noqa: E731

v.portada("Sobreajuste y validación cruzada", "Cuándo el árbol deja de aprender y empieza a memorizar",
          "Sexta parada, y la más importante de la clase. El árbol sobreajusta muchísimo. Hoy aprendes a medirlo sin engañarte, y a ver exactamente dónde empieza a memorizar.")

# ---- 1. Hook
v.b("El profesor fue enfático: esta metodología sobreajusta, de sobremanera. [[a]]Un árbol sin límites sigue partiendo hasta que cada hoja es pura. [[b]]Acierta el cien por ciento de sus propios datos, [[c]]y en datos nuevos no sirve para nada.",
    CL(), T("«sobreajusta de sobremanera»", 960, 230, tam=62, c="rojo", ent="subir"),
    P("sin límites: parte hasta hojas puras", 960, 400, "gris", 48, en="a"),
    tarjetas([(560, "verde", "100 %", "en sus propios datos", "b"), (1360, "rojo", "inútil", "en datos nuevos", "c")], y=650, w=640, h=280, tam_t=72),
    sec="1. El peligro")

# ---- 2. Analogía
v.b("Piensa en un estudiante que se prepara con una sola prueba de práctica. [[a]]En vez de entender la materia, memoriza las respuestas de esa prueba. [[b]]En la prueba de práctica saca nota perfecta. [[c]]En el examen real, con preguntas nuevas, reprueba. Eso es sobreajustar.",
    CL(), Bc(560, 480, 640, 400, "blanco", "verde", 5, 30, sombra=True, en="a", ent="subir"), T("prueba de práctica", 560, 330, tam=40, c="verde", en="a", t=0.1),
    T("respuestas memorizadas", 560, 420, tam=34, c="gris", w="sb", en="a", t=0.2), T("nota perfecta", 560, 540, tam=54, c="verde", en="b"), K(560, 620, "check", 60, en="b", t=0.3),
    Bc(1360, 480, 640, 400, "blanco", "rojo", 5, 30, sombra=True, en="c", ent="subir"), T("examen real", 1360, 330, tam=40, c="rojo", en="c", t=0.1),
    T("preguntas nuevas", 1360, 420, tam=34, c="gris", w="sb", en="c", t=0.2), T("reprueba", 1360, 540, tam=54, c="rojo", en="c", t=0.4), K(1360, 620, "cruz", 60, en="c", t=0.6),
    P("memorizar ≠ aprender", 960, 820, "violeta", 52, en="c", t=1.0), ilustrativo(1000, texto="analogía"), sec="2. Analogía: memorizar la prueba")

# ---- 3. Validación cruzada paso a paso
BX, BY, BW, BH = 520, 300, 190, 70
v.b("La defensa es medir sobre datos que el modelo no vio. [[a]]La validación cruzada parte a los tres mil ciento cincuenta clientes en cinco bloques de seiscientos treinta. [[b]]Entrena con cuatro bloques, dos mil quinientos veinte clientes, [[c]]y mide el AUC en el bloque restante, que el árbol nunca vio.",
    CL(), T(f"{mil(D['n'])} clientes → 5 bloques de {D['fold_n']}", 960, 190, tam=46, c="tinta", ent="subir", en="a"),
    *[[Bc(BX + 210 * j, BY, BW, BH, "azulclaro", "azul", 3, 14, en="a", t=0.12 * j, ent="pop", id=f"b0{j}"),
       T(f"bloque {j + 1}", BX + 210 * j, BY, tam=28, c="azul", w="sb", en="a", t=0.12 * j + 0.05, id=f"t0{j}")] for j in range(5)],
    *[U(f"b0{j}", relleno="verdeclaro", borde="verde", en="b", dur=0.4) for j in range(4)], U("b04", relleno="ambarclaro", borde="ambar", en="c", dur=0.4),
    T(f"entrena: {mil(D['fold_train'])}", 830, 400, tam=36, c="verde", w="b", en="b", t=0.3, id="lent"), T(f"mide: {D['fold_n']}", 1360, 400, tam=36, c="ambar", w="b", en="c", t=0.3, id="lmid"),
    sec="3. Validación cruzada")
v.b("Eso da un número. [[a]]Después se rota qué bloque mide, y se repite cinco veces. [[b]]Cada cliente sirve para medir exactamente una vez, y para entrenar cuatro.",
    *[[[Bc(BX + 210 * j, BY + 100 * r, BW, BH, "ambarclaro" if j == r else "verdeclaro", "ambar" if j == r else "verde", 3, 14, en="a", t=0.35 * (r - 1) + 0.04 * j, ent="pop"),
        T("mide" if j == r else "entrena", BX + 210 * j, BY + 100 * r, tam=26, c="ambar" if j == r else "verde", w="sb", en="a", t=0.35 * (r - 1) + 0.04 * j + 0.05)] for j in range(5)]
      for r in range(1, 5)],
    X("lent", "lmid", en="a", dur=0.2), U("b04", relleno="verdeclaro", borde="verde", en="a", dur=0.3), U("b00", relleno="ambarclaro", borde="ambar", en="a", dur=0.3),
    *[T(f"ronda {r + 1}", BX - 170, BY + 100 * r, tam=28, c="gris", w="sb", en=("a" if r else None), t=0.35 * max(r - 1, 0)) for r in range(5)],
    P("cada cliente: mide 1 vez · entrena 4", 960, 860, "teal", 44, en="b"))

# ---- 4. Ejemplo: las 5 rondas
v.b("Ahora los cinco resultados reales, con un árbol de profundidad cinco. [[a]]Ronda uno: cero coma nueve cero cero cinco. Ronda dos: cero coma nueve dos cero cuatro. [[b]]Ronda tres: cero coma nueve uno cero siete. Ronda cuatro: cero coma nueve uno siete cinco. [[c]]Y ronda cinco: cero coma ocho siete nueve nueve.",
    CL(), T("AUC de cada ronda · árbol de profundidad 5", 960, 180, tam=44, c="tinta", ent="subir"),
    *[[Bc(330 + 315 * i, 420, 280, 200, "blanco", "azul", 4, 24, sombra=True, en="abc"[min(i // 2, 2)], t=0.5 * (i % 2), ent="subir"),
       T(f"ronda {i + 1}", 330 + 315 * i, 360, tam=32, c="gris", w="sb", en="abc"[min(i // 2, 2)], t=0.5 * (i % 2) + 0.05),
       N(a, 330 + 315 * i, 440, "{:.4f}", 56, "azul", en="abc"[min(i // 2, 2)], t=0.5 * (i % 2) + 0.1, dur=0.6)] for i, a in enumerate(RO)],
    real(1050), sec="4. Ejemplo: las cinco rondas")
v.b("El promedio. [[a]]Suman cuatro coma cinco dos nueve. [[b]]Entre cinco: cero coma nueve cero cinco ocho. [[c]]Y el desvío entre rondas es cero coma cero uno cinco. El desvío importa tanto como el promedio: si las rondas dieran cero coma siete y cero coma nueve cinco, un promedio no significaría nada.",
    T(" + ".join(f"{a:.4f}" for a in RO) + f" = {sum(RO):.4f}", 960, 620, tam=36, c="tinta", w="sb", en="a"),
    T(f"{sum(RO):.4f} / 5 =", 900, 740, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(D["rondas_media"], 930, 740, "{:.4f}", 84, "violeta", anc="lm", en="b", t=0.4),
    P(f"desvío = {D['rondas_std']:.4f}", 960, 870, "ambar", 48, en="c"), real(1050))

# ---- 5. Estratificado y al azar
v.b("Dos detalles que no son detalles. [[a]]Primero, estratificar: cada bloque conserva el quince coma siete por ciento de fuga. Noventa y nueve fugados de seiscientos treinta, en los cinco bloques. [[b]]Al azar puro, algún bloque podría quedar casi sin fugados, y su AUC no se podría interpretar.",
    CL(), P("StratifiedKFold", 960, 200, "violeta", 52, ent="subir"),
    *[[Bc(330 + 315 * i, 450, 260, 180, "blanco", "teal", 4, 24, en="a", t=0.15 * i, ent="pop"), T(f"bloque {i + 1}", 330 + 315 * i, 400, tam=30, c="gris", w="sb", en="a", t=0.15 * i),
       T(pct(t_), 330 + 315 * i, 470, tam=44, c="teal", w="b", en="a", t=0.15 * i + 0.1)] for i, t_ in enumerate(D["fold_tasas"])],
    T(f"≈ {round(D['fold_tasas'][0] * D['fold_n'])} fugados de {D['fold_n']} en cada bloque", 960, 610, tam=40, c="tinta", w="sb", en="a", t=0.8),
    P("al azar puro: un bloque casi sin fugados", 960, 780, "rojo", 46, en="b"), real(1050), sec="5. Estratificado y al azar")
v.b("Segundo, el azar. [[a]]El profesor contó un error que ve en la práctica: tomar el entrenamiento de una fecha a otra, y probar con las fechas que quedan. [[b]]«Es un error gravísimo, pero pasa». La muestra de entrenamiento tiene que ser aleatoria. [[c]]Un matiz: cuando el modelo se va a usar en el futuro, además conviene una prueba fuera de tiempo, pero como chequeo extra, no en reemplazo.",
    CL(), T("partir por fecha sin querer", 960, 230, tam=56, c="tinta", en="a", id="pf"), U("pf", tachado=1.0, color="gris", en="b", dur=0.6),
    P("«es un error gravísimo, pero pasa»", 960, 390, "rojo", 50, en="b", t=0.3), P("entrenamiento: muestra aleatoria", 960, 530, "verde", 50, en="b", t=0.9),
    Bc(960, 760, 1400, 170, "blanco", "ambar", 4, 24, sombra=True, en="c", ent="subir"), T("matiz", 960, 715, tam=30, c="ambar", w="b", en="c", t=0.1),
    T("modelo para el futuro: además, una prueba fuera de tiempo (chequeo extra)", 960, 785, tam=34, c="tinta", w="sb", en="c", t=0.2))

# ---- 6. La curva
GX = dict(x0=260, y0=230, x1=1260, y1=840)
v.b("Ahora sí, la curva del sobreajuste. [[a]]Para cada profundidad máxima, dos números: el AUC en los mismos datos de entrenamiento, [[b]]y el AUC honesto, con validación cruzada.",
    CL(), G(GX["x0"], GX["y0"], GX["x1"], GX["y1"], -0.5, 9.5, 0.65, 1.02,
            [dict(xs=list(range(10)), ys=[c["tr"] for c in CU], color="gris", grosor=6, marcadores=9, revela=0.0),
             dict(xs=list(range(10)), ys=[c["va"] for c in CU], color="verde", grosor=7, marcadores=10, revela=0.0)],
            xticks=[], yticks=[0.7, 0.8, 0.9, 1.0], yfmt="{:.1f}", xlabel="", id="gp"),
    *[T(e, i, 0.625, tam=26, c="gris", w="sb", ref="gp") for i, e in enumerate(ETQ)], T("profundidad máxima del árbol", 760, 930, tam=32, c="tinta", w="sb"),
    T("AUC", GX["x0"] - 10, GX["y0"] - 40, tam=30, c="tinta", w="sb", anc="lm"),
    P("entrenamiento", 1560, 330, "gris", 44, en="a"), U("gp", **{"series.0.revela": 1.0}, en="a", t=0.3, dur=2.5),
    P("validación cruzada", 1560, 430, "verde", 44, en="b"), U("gp", **{"series.1.revela": 0.01}, en="b", dur=0.1), real(1050), sec="6. La curva del sobreajuste")
v.b("Mira las dos líneas. [[a]]La de entrenamiento sube siempre, hasta llegar a {1.000|uno}: el árbol termina sabiéndose de memoria a cada cliente. [[b]]La de validación sube hasta profundidad seis, [[c]]y después baja.",
    T(f"sin límite: {SL['tr']:.3f}", 1560, 560, tam=40, c="gris", w="b", en="a"),
    U("gp", **{"series.1.revela": ib / 9}, en="b", dur=2.0), Bc(ib, best["va"], 30, 30, None, "verde", 5, 15, ref="gp", en="b", t=2.0, ent="pop"),
    T(f"mejor: profundidad {best['d']} · {best['va']:.3f}", 1560, 650, tam=40, c="verde", w="b", en="b", t=1.8),
    U("gp", **{"series.1.revela": 1.0}, en="c", dur=1.6), T(f"sin límite: {SL['va']:.3f}", 1560, 740, tam=40, c="rojo", w="b", en="c", t=1.2))
v.b("Ese punto de quiebre es el ajuste adecuado del que hablaba la clase treinta y uno. [[a]]Sin límite, el árbol tiene doscientas veinticuatro hojas y veintiún niveles. [[b]]Y pasa de cero coma nueve uno uno a cero coma ocho dos siete: perdió más de ocho puntos de AUC por memorizar.",
    L(9, SL["va"], 9, SL["tr"], "rojo", 6, ref="gp", en="a"),
    P(f"{SL['hojas']} hojas · {SL['prof_real']} niveles", 1560, 840, "rojo", 40, en="a", t=0.3),
    P(f"{best['va']:.3f} → {SL['va']:.3f}: −{(best['va'] - SL['va']) * 100:.1f} puntos", 1560, 930, "rojo", 40, en="b"))

# ---- 7. Ejemplo: la brecha
v.b("Midamos la brecha entre entrenamiento y validación, en puntos de AUC. [[a]]Con profundidad tres: cero coma ocho cinco cinco menos cero coma ocho cinco cero, apenas medio punto. [[b]]Con profundidad seis, la mejor: cero coma nueve cuatro cero menos cero coma nueve uno uno, casi tres puntos. [[c]]Y sin límite: uno menos cero coma ocho dos siete, diecisiete puntos.",
    CL(), T("brecha = AUC entrenamiento − AUC validación", 960, 190, tam=46, c="violeta", ent="subir"),
    *[[Bc(960, 360 + 170 * k, 1400, 140, "blanco", c, 4, 24, sombra=True, en=m, ent="subir"),
       T(lab, 420, 360 + 170 * k, tam=38, c="tinta", w="sb", anc="lm", en=m, t=0.1),
       T(f"{cc['tr']:.3f} − {cc['va']:.3f} =", 1260, 360 + 170 * k, tam=40, c="gris", w="sb", anc="rm", en=m, t=0.2),
       N(brecha(cc), 1300, 360 + 170 * k, "{:.1f}", 60, c, suf=" pts", anc="lm", en=m, t=0.4)]
      for k, (lab, cc, c, m) in enumerate((("profundidad 3", c3, "verde", "a"), (f"profundidad {best['d']} (la mejor)", c6, "ambar", "b"), ("sin límite", SL, "rojo", "c")))],
    real(1050), sec="7. Ejemplo: la brecha")
v.b("El profesor dio su regla práctica: [[a]]como mucho unos dos puntos de diferencia entre entrenamiento y prueba. [[b]]Fíjate en el matiz: con esa regla, la profundidad cinco queda en dos coma dos, y la seis, que tiene el mejor AUC de validación, ya pasa los dos puntos. [[c]]No hay número mágico: es un criterio para decidir, no una ley.",
    CL(), P("«dos puntos a lo más»", 960, 240, "violeta", 60, ent="subir", en="a"),
    tarjetas([(560, "verde", f"prof. 5: {brecha(c5):.1f} pts", f"AUC validación {c5['va']:.3f}", "b"), (1360, "ambar", f"prof. 6: {brecha(c6):.1f} pts", f"AUC validación {c6['va']:.3f}", "b")],
             y=520, w=660, h=260, tam_t=50),
    P("un criterio para decidir, no una ley", 960, 790, "teal", 48, en="c"))

# ---- 8. ¿Demasiado bueno?
v.b("Y una tensión que conviene ver. [[a]]El profesor dijo que en la industria, un AUC de cero coma ocho uno a cero coma ocho cinco en fuga ya es un buen trabajo. [[b]]Y que si pasa de cero coma nueve, algo malo está pasando: quizás una variable del futuro. [[c]]Pero el árbol de la clase llega a cero coma nueve uno uno. [[d]]La explicación la dio él mismo: estos datos son de libro. En datos reales, un número así se revisa antes de celebrarlo.",
    CL(), alerta(160, texto="tensión del material"),
    tarjetas([(400, "verde", "0.81 – 0.85", "industria: buen trabajo", "a"), (960, "rojo", "> 0.90", "«algo malo está pasando»", "b"),
              (1520, "ambar", f"{best['va']:.3f}", "el árbol de la clase", "c")], y=470, w=480, h=300, tam_t=64),
    P("«son datos de libro»: en datos reales, revisa antes de celebrar", 960, 790, "violeta", 42, en="d"), sec="8. ¿Demasiado bueno?")

# ---- 9. Error típico
v.b("El error típico de esta parte: [[a]]elegir la profundidad mirando el AUC de entrenamiento, o mirando el test. [[b]]La complejidad se elige con validación cruzada sobre el entrenamiento, y el test se toca una sola vez, al final.",
    CL(), T("elegir mirando el entrenamiento… o el test", 960, 300, tam=56, c="tinta", en="a", id="e1"), U("e1", tachado=1.0, color="gris", en="b", dur=0.6),
    P("elige con validación cruzada", 960, 480, "verde", 54, en="b", t=0.4), P("el test se toca una sola vez, al final", 960, 640, "violeta", 50, en="b", t=1.0),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tu árbol logra cero coma noventa y nueve en entrenamiento, y cero coma setenta y uno en prueba. [[b]]¿Qué haces, y en qué orden?",
    CL(), P("0.99 en entrenamiento · 0.71 en prueba", 960, 320, "rojo", 56, en="a"),
    T("¿qué haces, y en qué orden?", 960, 520, tam=66, c="tinta", en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Primero, regularizar: bajar la profundidad y subir el mínimo de clientes por hoja, mirando la curva con validación cruzada. [[a]]Después, podar con un alfa validado. [[b]]Y solo si todavía no alcanza, pasar a un bosque. [[c]]La respuesta incompleta es «cambio a Random Forest» sin diagnosticar.",
    CL(), *[[Bc(400 + 560 * i, 450, 500, 260, "blanco", c, 5, 30, sombra=True, en=m, ent="subir"), T(str(i + 1), 400 + 560 * i - 190, 450, tam=80, c=c, en=m, t=0.1),
             T(txt, 400 + 560 * i + 40, 450, tam=36, c="tinta", w="sb", mw=360, en=m, t=0.2)]
            for i, (txt, c, m) in enumerate((("regularizar: profundidad y hojas mínimas", "verde", None), ("podar con alfa validado", "violeta", "a"), ("solo entonces, un bosque", "azul", "b")))],
    P("incompleto: «cambio a Random Forest» sin diagnosticar", 960, 760, "rojo", 44, en="c"), T("pregunta 5 de la autoevaluación (Arbol de decision.pptx)", 960, 870, tam=28, c="gris", w="sb", en="c", t=0.5))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el AUC de entrenamiento siempre sube, el de validación sube y después baja. [[b]]Se elige la profundidad donde la validación es máxima, aquí seis. [[c]]Y la validación cruzada es aleatoria, estratificada, y se mira el promedio y el desvío.",
    CL(), P("entrenamiento sube siempre · validación sube y baja", 960, 300, "violeta", 48, en="a"),
    P(f"elige el máximo de validación: profundidad {best['d']} ({best['va']:.3f})", 960, 480, "verde", 48, en="b"),
    P("CV: aleatoria, estratificada, promedio y desvío", 960, 660, "teal", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]cómo frenar al árbol. Antes de crecer, o después: la poda.",
    CL(), siguiente(7, "Poda: antes o después de crecer"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
