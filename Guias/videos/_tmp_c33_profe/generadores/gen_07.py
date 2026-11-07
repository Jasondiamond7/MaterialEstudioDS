import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, alerta

OUT = out("guion_c33_07_topk.json")
v = V("Cuatro contra cuatro: el ranking ingenuo")
T4, M4, KT, SM = D["top4"], D["manual4"], D["k_tabla"], D["semillas"]
PT, PM = D["pares_top4"], D["pares_manual4"]
S6, H6, N6 = D["slide_top6"], D["slide_man6"], D["top6_nb"]

v.portada("Cuatro contra cuatro", "Mismo modelo, mismos datos, mismo número de variables: solo cambia cómo se eligen",
          "Sexta parada: el clímax de la clase. Cuatro variables contra cuatro variables.")

# ---- 1. Hook
v.b("¿Te acuerdas del entrenador del primer video? [[a]]Elegía a los once que mejor jugaban solos, y terminaba con cuatro arqueros. [[b]]Hoy lo vamos a ver pasar con variables reales, y con números.",
    CL(), Pt([(7, [560, 420, 1100, 520], "azul"), (4, [1180, 420, 1360, 520], "ambar")], 18, visibles=11, dur=1.0),
    P("4 arqueros", 1270, 600, "ambar", 44, en="a"), T("cada uno bueno · el equipo malo", 960, 300, tam=52, c="gris", en="a", t=0.4),
    P("hoy, con variables reales", 960, 780, "violeta", 52, en="b"), ilustrativo(1000, texto="analogía"), sec="1. El equipo de cuatro arqueros")

# ---- 2. Los dos equipos
v.b("El experimento: el mismo modelo, los mismos datos, y el mismo número de variables. Cuatro. [[a]]Equipo uno: las cuatro primeras del ranking de información mutua. Segundos, logaritmo de segundos, contactos por mes, y logaritmo del valor del cliente. [[b]]Equipo dos: cuatro elegidas a mano, evitando redundancia. Reclamos, segundos, estado dos, y número de fallas.",
    CL(), T("mismo modelo · mismos datos · 4 variables", 960, 190, tam=44, c="gris", w="sb", ent="subir"),
    Bc(520, 560, 720, 560, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("top-4 por información mutua", 520, 340, tam=40, c="rojo", en="a", t=0.1),
    *[P(nm, 520, 440 + 100 * i, "rojo", 40, en="a", t=0.5 + 0.45 * i) for i, nm in enumerate(T4["cols"])],
    Bc(1400, 560, 720, 560, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("4 sin redundancia (a mano)", 1400, 340, tam=40, c="verde", en="b", t=0.1),
    *[P(nm, 1400, 440 + 100 * i, "verde", 40, en="b", t=0.5 + 0.45 * i) for i, nm in enumerate(M4["cols"])], sec="2. Los dos equipos")
v.b("Antes de medir, mira el equipo uno. [[a]]Segundos y su logaritmo son la misma información con otra escala: una es función exacta de la otra. [[b]]Su correlación de Pearson es solo cero coma seis tres cuatro, porque el log curva la relación. Pero no aportan nada nuevo entre sí.",
    R(520, 490, 600, 200, "rojo", 30, en="a"), T("misma información", 140, 490, tam=30, c="rojo", w="sb", mw=180, en="a", t=0.4),
    P(f"Pearson = {PT[0][2]:.3f} (el log curva la relación)", 960, 930, "ambar", 40, en="b"))

# ---- 3. Ejemplo: correlación media del equipo
v.b("Midamos la redundancia de cada equipo: la correlación media entre sus seis pares. [[a]]Equipo uno: cero coma seis tres cuatro, cero coma cinco cero cinco, cero coma cinco dos cuatro, cero coma cuatro seis, cero coma siete nueve ocho, y cero coma tres nueve tres. [[b]]Suman tres coma tres uno cuatro. Entre seis: cero coma cincuenta y cinco.",
    CL(), T("correlación media entre los 6 pares · equipo 1", 960, 180, tam=40, c="rojo", ent="subir"),
    *[[Bc(960, 270 + 72 * i, 1300, 60, "blanco", "linea", 2, 14, en="a", ent="subir", t=0.5 * i), T(f"{a}  ·  {b}", 360, 270 + 72 * i, tam=32, c="tinta", w="sb", anc="lm", en="a", t=0.5 * i + 0.05),
       T(f"{r:.3f}", 1560, 270 + 72 * i, tam=36, c="rojo", w="b", anc="rm", en="a", t=0.5 * i + 0.1)] for i, (a, b, r) in enumerate(PT)],
    T("0.634 + 0.505 + 0.524 + 0.460 + 0.798 + 0.393 = 3.314", 960, 740, tam=40, c="tinta", w="sb", en="b"),
    T("3.314 / 6 =", 900, 840, tam=48, c="tinta", en="b", t=0.6, anc="rm"), N(T4["cm"], 930, 840, "{:.2f}", 72, "rojo", en="b", t=0.9, anc="lm"),
    sec="3. Ejemplo: la redundancia de cada equipo")
v.b("Equipo dos. [[a]]Los seis pares suman uno coma tres dos cuatro. [[b]]Entre seis: cero coma veintidós. Menos de la mitad de redundancia.",
    CL(), T("correlación media entre los 6 pares · equipo 2", 960, 180, tam=40, c="verde", ent="subir"),
    *[[Bc(960, 270 + 72 * i, 1300, 60, "blanco", "linea", 2, 14, ent="subir", t=0.12 * i), T(f"{a}  ·  {b}", 360, 270 + 72 * i, tam=32, c="tinta", w="sb", anc="lm", t=0.12 * i + 0.05),
       T(f"{r:.3f}", 1560, 270 + 72 * i, tam=36, c="verde", w="b", anc="rm", t=0.12 * i + 0.1)] for i, (a, b, r) in enumerate(PM)],
    T("0.123 + 0.179 + 0.103 + 0.304 + 0.497 + 0.118 = 1.324", 960, 740, tam=40, c="tinta", w="sb", en="a"),
    T("1.324 / 6 =", 900, 840, tam=48, c="tinta", en="b", anc="rm"), N(M4["cm"], 930, 840, "{:.2f}", 72, "verde", en="b", t=0.3, anc="lm"))

# ---- 4. El resultado
v.b("Ahora sí, el resultado, con validación cruzada de cinco folds. [[a]]El top cuatro por información mutua: AUC cero coma ocho cero cuatro. [[b]]Las cuatro sin redundancia: cero coma ocho nueve tres. [[c]]Casi nueve puntos de AUC de diferencia, con la misma cantidad de variables.",
    CL(), T("AUC · validación cruzada de 5 folds · base real", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    Bx(440, 860 - (T4["auc"] - 0.7) * 2400, 760, 860, "rojo", None, 0, 12, grow="y", en="a"), N(T4["auc"], 600, 860 - (T4["auc"] - 0.7) * 2400 - 48, "{:.4f}", 70, "rojo", en="a", t=0.3, dur=0.9),
    T(f"top-4 por MI · corr. media {T4['cm']:.2f}", 600, 910, tam=32, c="rojo", w="sb", en="a"),
    Bx(1160, 860 - (M4["auc"] - 0.7) * 2400, 1480, 860, "verde", None, 0, 12, grow="y", en="b"), N(M4["auc"], 1320, 860 - (M4["auc"] - 0.7) * 2400 - 48, "{:.4f}", 70, "verde", en="b", t=0.3, dur=0.9),
    T(f"4 sin redundancia · corr. media {M4['cm']:.2f}", 1320, 910, tam=32, c="verde", w="sb", en="b"),
    L(340, 860, 1580, 860, "tinta", 4), T("eje desde 0.70", 1700, 860, tam=26, c="gris", w="sb"),
    P(f"+{(M4['auc'] - T4['auc']) * 100:.1f} puntos de AUC", 960, 640, "violeta", 50, en="c"), sec="4. El resultado")
v.b("El mecanismo está a la vista. [[a]]El top cuatro gasta dos de sus cuatro cupos en la misma variable. [[b]]Y no incluye nada de reclamos, que es el predictor individual más fuerte de la base, con correlación cero coma cincuenta y uno. [[c]]Cada variable está bien elegida por separado. El conjunto es malo.",
    CL(), T("¿por qué pierde el top-4?", 960, 200, tam=48, c="tinta", ent="subir"),
    P("2 de 4 cupos: segundos y log_segundos", 960, 360, "rojo", 48, en="a"),
    P(f"ningún reclamo (r con fuga = {D['corr_fuga']['reclamos']:.3f}, el más fuerte)", 960, 520, "rojo", 46, en="b"),
    P("cada una bien elegida · el conjunto, malo", 960, 720, "violeta", 54, en="c"))

# ---- 5. Dos verificaciones
v.b("Un número que aparece una sola vez no es un resultado. Primera verificación: [[a]]¿depende de la semilla con que se estima la información mutua? Con cinco semillas distintas, [[b]]el AUC del top cuatro se mueve entre cero coma ocho cero cuatro y cero coma ocho uno cero. [[c]]Y en las cinco, segundos y su logaritmo ocupan dos cupos, y no hay ningún reclamo.",
    CL(), T("verificación 1 · cinco semillas de la información mutua", 960, 190, tam=40, c="tinta", ent="subir"),
    *[[Bc(960, 320 + 100 * i, 1500, 84, "blanco", "linea", 2, 18, en="a", ent="subir", t=0.3 * i), T(f"semilla {s_}", 330, 320 + 100 * i, tam=32, c="gris", w="sb", anc="lm", en="a", t=0.3 * i),
       T(" · ".join(cols), 560, 320 + 100 * i, tam=28, c="tinta", w="sb", anc="lm", en="a", t=0.3 * i + 0.1),
       T(f"{a:.4f}", 1690, 320 + 100 * i, tam=36, c="rojo", w="b", anc="rm", en="b", t=0.2 * i)] for i, (s_, a, cols) in enumerate(SM)],
    P("siempre segundos + log_segundos · nunca reclamos", 960, 860, "ambar", 44, en="c"), sec="5. Dos verificaciones")
GK = dict(x0=300, y0=280, x1=1180, y1=840)
v.b("Segunda verificación: ¿qué pasa cuando crece el presupuesto de variables? [[a]]La clase compara el top k contra una selección automática que evita redundancia, para k de cuatro a doce. [[b]]Con cuatro, evitar redundancia gana siete puntos: cero coma ocho siete cinco contra cero coma ocho cero cuatro. [[c]]Pero con seis o más, la ventaja se da vuelta.",
    CL(), T("verificación 2 · AUC según el número de variables k", 960, 190, tam=40, c="tinta", ent="subir"),
    G(GK["x0"], GK["y0"], GK["x1"], GK["y1"], 3.5, 12.5, 0.78, 0.92,
      [dict(xs=[r[0] for r in KT], ys=[r[1] for r in KT], color="rojo", grosor=6, marcadores=10, revela=0.0),
       dict(xs=[r[0] for r in KT], ys=[r[2] for r in KT], color="verde", grosor=6, marcadores=10, revela=0.0)],
      xticks=[4, 6, 8, 10, 12], yticks=[0.78, 0.82, 0.86, 0.90], yfmt="{:.2f}", xlabel="k = número de variables", id="gk", en="a"),
    P("top-k por MI", 1450, 360, "rojo", 40, en="a", t=0.4), P("evitando redundancia", 1450, 450, "verde", 40, en="a", t=0.6),
    U("gk", **{"series.0.revela": 0.01, "series.1.revela": 0.01}, en="b", dur=0.1),
    T(f"k = 4: {KT[0][2]:.3f} vs {KT[0][1]:.3f}", 1450, 600, tam=38, c="verde", w="sb", en="b", t=0.3),
    U("gk", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="c", dur=2.0),
    T(f"k = 6: {KT[1][2]:.3f} vs {KT[1][1]:.3f}", 1450, 690, tam=38, c="rojo", w="sb", en="c", t=0.6),
    T(f"k = 12: {KT[4][2]:.3f} vs {KT[4][1]:.3f}", 1450, 780, tam=38, c="rojo", w="sb", en="c", t=1.4))
v.b("La lección no es «los filtros son malos». Es más precisa: [[a]]el costo de ignorar la redundancia crece cuando el presupuesto de variables es chico. [[b]]Si te quedas con treinta de cincuenta, el ranking alcanza. Si tienes que llegar a cinco porque el modelo va a un comité, no.",
    CL(), T("la lección precisa", 960, 210, tam=48, c="tinta", ent="subir"),
    P("presupuesto chico → la redundancia duele", 960, 380, "rojo", 54, en="a"),
    tarjetas([(560, "verde", "30 de 50", "el ranking alcanza", "b"), (1360, "rojo", "5 para un comité", "hay que evitar redundancia", "b")], y=650, w=640, h=300))

# ---- 6. Alerta del material: 6 contra 6
v.b("Ahora, una alerta del material. [[a]]La diapositiva veintitrés compara seis contra seis: el top seis da cero coma ocho uno cero cinco, y seis elegidas a mano, cero coma nueve cero cuatro uno. [[b]]Los reproduje con la base: las dos cifras son correctas, para esas dos listas.",
    CL(), alerta(),
    *[[Bc(x, 450, 640, 280, "blanco", c, 5, 30, sombra=True, en="a", ent="subir", t=t_), T(lab, x, 360, tam=36, c=c, w="sb", en="a", t=t_ + 0.1),
       N(a, x, 450, "{:.4f}", 84, c, en="a", t=t_ + 0.2), T(f"corr. media {cm:.2f}", x, 530, tam=32, c="gris", w="sb", en="a", t=t_ + 0.3)]
      for x, lab, a, cm, c, t_ in ((560, "top-6 de la diapositiva", S6["auc"], S6["cm"], "rojo", 0.0), (1360, "6 a mano de la diapositiva", H6["auc"], H6["cm"], "verde", 0.5))],
    P("reproducidas con la base: correctas para esas listas", 960, 720, "teal", 44, en="b"), K(1720, 720, "check", 50, en="b", t=0.6), sec="6. Alerta: seis contra seis")
v.b("Pero el top seis de la diapositiva no es el del notebook. [[a]]En la diapositiva, el sexto lugar es llamadas por mes. En el notebook es tiene reclamo. [[b]]Las dos tenían información mutua casi empatada, cero coma uno cero dos tres en la diapositiva. [[c]]Con tiene reclamo dentro, el top seis sube a cero coma ocho nueve siete. Y la ventaja de elegir a mano baja de nueve puntos a menos de uno.",
    CL(), T("el 6.º lugar decide", 960, 200, tam=48, c="tinta", ent="subir"),
    *[[Bc(x, 430, 640, 220, "blanco", c, 5, 26, sombra=True, en="a", ent="subir", t=t_), T(lab, x, 380, tam=34, c=c, w="sb", en="a", t=t_ + 0.1),
       T(nm, x, 460, tam=44, c="tinta", w="m", en="a", t=t_ + 0.2)] for x, lab, nm, c, t_ in ((560, "diapositiva · 6.º", "llamadas_por_mes", "gris", 0.0), (1360, "notebook · 6.º", "tiene_reclamo", "violeta", 0.5))],
    P("MI casi empatada en la 4.ª cifra decimal", 960, 600, "ambar", 44, en="b"),
    T(f"top-6 del notebook: {N6['auc']:.3f}  ·  6 a mano: {H6['auc']:.3f}  ·  diferencia {(H6['auc'] - N6['auc']) * 100:.1f} puntos", 960, 750, tam=40, c="tinta", w="sb", en="c"),
    P("la conclusión «9 puntos con 6 variables» depende de un empate", 960, 870, "rojo", 42, en="c", t=1.0))

# ---- 7. Error típico
v.b("El error típico, entonces: [[a]]elegir el top k de un ranking univariado cuando el presupuesto es chico, sin mirar la correlación entre las elegidas. [[b]]La corrección es simple: antes de aceptar el top, revisa la matriz de correlación de las que elegiste.",
    CL(), T("top-k univariado con k chico", 960, 300, tam=70, c="tinta", en="a", id="tk"), U("tk", tachado=1.0, color="gris", en="b", t=0.1, dur=0.6),
    P("revisa la correlación entre las elegidas", 960, 520, "verde", 56, en="b", t=0.4), T("X[elegidas].corr().abs()", 960, 680, tam=46, c="tinta", w="m", en="b", t=1.0),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con k igual a doce, el top doce por información mutua da cero coma nueve cero uno, y la selección sin redundancia, cero coma ocho ocho seis. [[b]]¿Por qué ahora gana el ranking?",
    CL(), T(f"k = 12:  top-k {KT[4][1]:.3f}  ·  sin redundancia {KT[4][2]:.3f}", 960, 340, tam=54, c="violeta", en="a"),
    T("¿por qué ahora gana el ranking?", 960, 540, tam=64, c="tinta", en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Porque con doce cupos, el ranking termina incluyendo por su cuenta a las variables fuertes, como tiene reclamo y reclamo por antigüedad. [[a]]Los cupos repetidos ya no le cuestan tanto. [[b]]La redundancia duele cuando sobran pocas sillas.",
    CL(), P("con 12 cupos, las fuertes entran igual", 960, 300, "teal", 56, ent="subir"),
    T("tiene_reclamo (6.º) · reclamo_x_antig (9.º)", 960, 460, tam=46, c="tinta", w="sb", en="a"), P("la redundancia duele con pocas sillas", 960, 650, "violeta", 54, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]cuatro contra cuatro, el ranking ingenuo pierde casi nueve puntos de AUC, porque gasta cupos en redundantes. [[b]]Es estable ante la semilla, pero se diluye cuando crece k. [[c]]Y un ranking por información mutua puede cambiar por un empate en la cuarta cifra.",
    CL(), P(f"top-4 {T4['auc']:.3f} vs sin redundancia {M4['auc']:.3f}", 960, 300, "rojo", 52, en="a"),
    P("estable con la semilla · se diluye al crecer k", 960, 480, "teal", 52, en="b"), P("un empate en la 4.ª cifra cambia el top-6", 960, 660, "ambar", 50, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]los métodos que sí miran el conjunto. Wrapper y embedded: RFE y Lasso.",
    CL(), siguiente(8, "Wrapper y embedded: RFE y Lasso"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
