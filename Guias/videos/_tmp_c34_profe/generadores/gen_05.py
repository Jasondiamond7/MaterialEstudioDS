import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, NODOS

OUT = out("guion_c34_05_barrido.json")
v = V("El barrido voraz")
BR = D["barrido"]
N2R, G2R = D["nivel2_reclamo"]
N2N, G2N = D["nivel2_sin_reclamo"]
ND = NODOS


def g2(f, n):
    p = f / n
    return 1 - p * p - (1 - p) ** 2


# numero_fallas en el nodo 9 (reclamo y usa poco): ganancia local
n9, n10, n11 = ND[9], ND[10], ND[11]
g9, g10, g11 = g2(n9["fugas"], n9["n"]), g2(n10["fugas"], n10["n"]), g2(n11["fugas"], n11["n"])
w10, w11 = n10["n"] / n9["n"], n11["n"] / n9["n"]
gp9 = w10 * g10 + w11 * g11
gan9 = g9 - gp9
gan_fallas_raiz = next(b["gan"] for b in BR if b["var"] == "numero_fallas")

v.portada("El barrido voraz", "Cientos de cortes para elegir una pregunta, y por qué el árbol no mira adelante",
          "Quinta parada. El árbol no prueba dos candidatos: prueba cientos, en cada nodo. Hoy ves el barrido completo, y entiendes por qué se le llama voraz.")

# ---- 1. Hook
v.b("En el video anterior comparaste dos cortes a mano. [[a]]Para elegir la primera pregunta, el notebook de la clase evalúa trescientas setenta y ocho combinaciones de variable y umbral. [[b]]Y ese barrido completo se repite en cada nodo del árbol.",
    CL(), T("para elegir UNA pregunta", 960, 260, tam=56, c="tinta", ent="subir"),
    N(D["combos"], 960, 450, "{:.0f}", 180, "violeta", en="a", dur=1.6), T("combinaciones de variable y umbral", 960, 590, tam=44, c="gris", w="sb", en="a", t=0.5),
    P("y se repite en CADA nodo", 960, 760, "rojo", 52, en="b"), real(1050), sec="1. Cientos de cortes")

# ---- 2. Los umbrales candidatos
v.b("¿De dónde salen los umbrales? [[a]]Se ordenan los valores distintos de la variable. [[b]]Y se prueba un corte en el punto medio entre cada par de valores consecutivos. [[c]]Por ejemplo, con edades de veintidós, veinticinco, treinta y uno y cuarenta, los cortes candidatos son veintitrés coma cinco, veintiocho, y treinta y cinco coma cinco.",
    CL(), T("umbrales candidatos de una variable", 960, 200, tam=48, c="tinta", ent="subir"),
    L(300, 520, 1620, 520, "tinta", 5, en="a"),
    *[[Bc(300 + (x - 20) * 60, 520, 34, 34, "azul", None, 0, 17, en="a", t=0.2 * i), T(str(x), 300 + (x - 20) * 60, 590, tam=40, c="azul", w="b", en="a", t=0.2 * i)]
      for i, x in enumerate((22, 25, 31, 40))],
    *[[L(300 + (m - 20) * 60, 420, 300 + (m - 20) * 60, 500, "violeta", 6, en="c", t=0.4 * i), T(f"{m:g}", 300 + (m - 20) * 60, 380, tam=40, c="violeta", w="b", en="c", t=0.4 * i + 0.1)]
      for i, m in enumerate((23.5, 28, 35.5))],
    T("puntos medios entre valores consecutivos", 960, 700, tam=40, c="violeta", w="sb", en="b"),
    T("k valores distintos → k − 1 cortes", 960, 790, tam=40, c="gris", w="sb", en="b", t=0.5),
    ilustrativo(1000, texto="ejemplo de las notas de la diapositiva 9 (Arbol de decision.pptx)"), sec="2. Los umbrales candidatos")
v.b("Segundos tiene mil setecientos cuarenta y cinco valores distintos: serían mil setecientos cuarenta y cuatro cortes, solo para esa variable. [[a]]El notebook de la clase simplifica: prueba sesenta percentiles, entre el dos y el noventa y ocho por ciento. [[b]]Y descarta cualquier corte que deje menos de sesenta clientes en un lado, el mismo criterio que min_samples_leaf. [[c]]Así salen las trescientas setenta y ocho combinaciones.",
    CL(), T(f"segundos: {mil(D['nunique']['segundos'])} valores distintos", 960, 220, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "violeta", "60 percentiles", "del 2 % al 98 % de cada variable", "a"), (1360, "ambar", "≥ 60 por lado", "igual que min_samples_leaf = 60", "b")], y=520, w=640, h=300),
    P(f"12 variables → {D['combos']} combinaciones para la raíz", 960, 800, "teal", 48, en="c"))

# ---- 3. El ranking de la raíz
X0, Y0, DY, ESC = 600, 250, 62, 13000
v.b("Este es el resultado del barrido en la raíz: el mejor corte de cada variable, ordenado por ganancia. [[a]]Reclamos, arriba de todo, con cero coma cero seis nueve tres. [[b]]Después vienen estado, frecuencia de llamadas, ce ele ve y segundos, todas cerca de cero coma cero dos seis. [[c]]Y al fondo, número de fallas y edad: casi nada.",
    CL(), T("mejor corte de cada variable en la raíz · ganancia de Gini", 960, 170, tam=40, c="tinta", ent="subir"),
    *[[T(b["var"], X0 - 20, Y0 + DY * i, tam=32, c="tinta", w="sb", anc="rm"),
       Bx(X0, Y0 + DY * i - 20, X0 + max(b["gan"] * ESC, 4), Y0 + DY * i + 20, "rojo" if i == 0 else ("gris" if i < 9 else "ambar"), None, 0, 8, grow="x",
          en=("a" if i == 0 else ("b" if i < 9 else "c")), t=0.08 * i),
       T(f"{b['gan']:.4f}", X0 + max(b["gan"] * ESC, 4) + 16, Y0 + DY * i, tam=30, c="rojo" if i == 0 else "gris", w="b", anc="lm",
         en=("a" if i == 0 else ("b" if i < 9 else "c")), t=0.08 * i + 0.3)]
      for i, b in enumerate(BR)],
    real(1050), sec="3. El ranking de la raíz")
v.b("¿Cuánto más gana reclamos? [[a]]Cero coma cero seis nueve tres, dividido en cero coma cero dos siete: [[b]]dos coma seis veces la segunda. No hay competencia. [[c]]Por eso reclamos queda en la raíz en todas las corridas de la clase.",
    CL(), T(f"{BR[0]['gan']:.4f} / {BR[1]['gan']:.4f} =", 900, 360, tam=76, c="tinta", w="sb", anc="rm", en="a"),
    N(BR[0]["gan"] / BR[1]["gan"], 930, 360, "{:.1f}", 110, "rojo", suf=" ×", anc="lm", en="b", dur=1.0),
    P("más del doble que la segunda: no hay competencia", 960, 560, "rojo", 48, en="b", t=0.7),
    P("reclamos = raíz en todas las corridas", 960, 700, "violeta", 48, en="c"),
    T("la diapositiva 12 muestra 9 de las 12 variables (omite frec_sms, monto y tipo_plan)", 960, 880, tam=30, c="gris", w="sb", en="c", t=0.8),
    sec="4. Ejemplo: más del doble")

# ---- 5. Voraz
v.b("Ahora la palabra clave: voraz. [[a]]Imagina a un excursionista en la niebla que quiere llegar a la cumbre. [[b]]En cada paso, sube por donde la pendiente es más empinada. [[c]]Avanza rápido, pero puede terminar en un cerro chico, sin saber que la montaña alta estaba al lado.",
    CL(), G(260, 300, 1320, 860, 0, 10, 0, 10,
            [dict(xs=[0, 1, 2, 3, 3.8, 4.6, 5.5, 6.5, 7.5, 8.5, 9.5, 10], ys=[1, 2.4, 4.2, 5.4, 5.8, 5.1, 4.6, 6.4, 8.6, 9.6, 8.3, 7], color="verde", grosor=8, suave=True, revela=1.0)],
      xticks=[], yticks=[], id="mt", en="a", rejilla=False),
    Bc(1.0, 2.4, 26, 26, "violeta", None, 0, 13, ref="mt", en="b", id="ex"), U("ex", x=2.0, y=4.2, en="b", t=0.4, dur=0.6), U("ex", x=3.0, y=5.4, en="b", t=1.0, dur=0.6), U("ex", x=3.8, y=5.8, en="b", t=1.6, dur=0.6),
    P("cerro chico", 3.8, 7.3, "ambar", 40, en="c", ref="mt"), P("la cumbre", 8.5, 10.9, "verde", 40, en="c", t=0.4, ref="mt"),
    ilustrativo(1000, texto="analogía"), sec="5. Por qué se dice voraz")
v.b("El árbol es igual. [[a]]En cada nodo elige el mejor corte ahora, sin mirar dos pasos adelante. [[b]]Y nunca vuelve atrás a revisar un corte anterior. [[c]]Por eso no garantiza el árbol óptimo: solo uno bueno, y rápido de construir.",
    CL(), P("elige lo mejor AHORA", 960, 280, "violeta", 60, en="a"), P("nunca revisa un corte anterior", 960, 440, "gris", 54, en="b"),
    tarjetas([(560, "rojo", "no garantiza", "el árbol óptimo", "c"), (1360, "verde", "sí garantiza", "uno bueno y rápido", "c")], y=700, w=640, h=260))

# ---- 6. El segundo nivel
v.b("Una vez elegido reclamos, el algoritmo repite el barrido completo dentro de cada hijo, por separado. [[a]]Entre los que reclamaron, el Gini del nodo es cero coma tres siete. [[b]]Frecuencia de llamadas gana con claridad: cero coma cero siete uno, el doble que segundos.",
    CL(), T("segundo nivel · barrido dentro de cada hijo", 960, 160, tam=44, c="tinta", ent="subir"),
    Bc(960, 520, 1300, 520, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"),
    T(f"hijo «reclamó»: {ND[8]['n']} clientes · Gini {g2(ND[8]['fugas'], ND[8]['n']):.3f}", 960, 320, tam=40, c="rojo", en="a", t=0.1),
    *[[T(var, 760, 420 + 90 * i, tam=36, c="tinta", w="sb", anc="rm", en="b", t=0.2 * i),
       Bx(780, 400 + 90 * i, 780 + g * 8000, 440 + 90 * i, "rojo" if i == 0 else "gris", None, 0, 8, grow="x", en="b", t=0.2 * i),
       T(f"{g:.4f}", 800 + g * 8000, 420 + 90 * i, tam=32, c="rojo" if i == 0 else "gris", w="b", anc="lm", en="b", t=0.2 * i + 0.3)]
      for i, (var, u, g) in enumerate(N2R)],
    real(1050), sec="6. El segundo nivel")
v.b("Entre los que no reclamaron, en cambio, pasa algo interesante. [[a]]Segundos gana con cero coma cero dos tres dos. [[b]]Pero frecuencia de llamadas da cero coma cero dos dos seis, y ce ele ve cero coma cero dos dos uno. [[c]]Es casi un empate: una diferencia de seis diezmilésimas. Guarda este dato para el video nueve.",
    CL(), T("segundo nivel · barrido dentro de cada hijo", 960, 160, tam=44, c="tinta"),
    Bc(960, 520, 1300, 520, "blanco", "verde", 5, 30, sombra=True, ent="subir"),
    T(f"hijo «no reclamó»: {mil(ND[1]['n'])} clientes · Gini {g2(ND[1]['fugas'], ND[1]['n']):.3f}", 960, 320, tam=40, c="verde", t=0.1),
    *[[T(f"{var} ≤ {u:g}", 760, 420 + 90 * i, tam=34, c="tinta", w="sb", anc="rm", en=("a" if i == 0 else "b"), t=0.25 * i),
       Bx(780, 400 + 90 * i, 780 + g * 20000, 440 + 90 * i, "verde" if i == 0 else "ambar", None, 0, 8, grow="x", en=("a" if i == 0 else "b"), t=0.25 * i),
       T(f"{g:.4f}", 800 + g * 20000, 420 + 90 * i, tam=32, c="tinta", w="b", anc="lm", en=("a" if i == 0 else "b"), t=0.25 * i + 0.3)]
      for i, (var, u, g) in enumerate(N2N)],
    P(f"casi empate: {N2N[0][2] - N2N[1][2]:.4f} de diferencia", 960, 860, "ambar", 46, en="c"), real(1050))

# ---- 7. Ejemplo: numero_fallas
v.b("Y mira lo que pasa con número de fallas. [[a]]En la raíz ganaba apenas cero coma cero cero uno uno: casi nada. [[b]]Pero en el tercer nivel, entre los que reclamaron y usan poco, separa noventa y cuatro por ciento de fuga de sesenta y siete. Calculemos su ganancia ahí.",
    CL(), T("numero_fallas", 960, 180, tam=60, c="tinta", ent="subir"),
    Bc(560, 450, 640, 260, "blanco", "gris", 5, 30, sombra=True, en="a", ent="subir"), T("en la raíz", 560, 380, tam=40, c="gris", w="sb", en="a", t=0.1),
    N(gan_fallas_raiz, 560, 470, "{:.4f}", 80, "gris", en="a", t=0.3),
    Bc(1360, 450, 640, 260, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("reclamó y usa poco", 1360, 380, tam=40, c="rojo", w="sb", en="b", t=0.1),
    T(f"{pct(n10['fugas'] / n10['n'])}  vs  {pct(n11['fugas'] / n11['n'])}", 1360, 470, tam=54, c="rojo", w="b", en="b", t=0.3),
    real(1050), sec="7. Ejemplo: número de fallas")
v.b("El nodo tiene doscientos dieciséis clientes, con Gini cero coma dos cuatro seis. [[a]]Los sin fallas son sesenta y siete, con Gini cero coma cuatro cuatro uno. Los con fallas, ciento cuarenta y nueve, con Gini cero coma uno uno cuatro. [[b]]Ponderado: cero coma dos uno cinco. [[c]]Ganancia: cero coma cero tres uno. Veintiocho veces lo que ganaba en la raíz.",
    CL(), T(f"nodo: {n9['n']} clientes · Gini {g9:.3f}", 960, 190, tam=46, c="tinta", ent="subir"),
    T(f"sin fallas: {n10['n']} · Gini {g10:.3f}      con fallas: {n11['n']} · Gini {g11:.3f}", 960, 320, tam=40, c="gris", w="sb", en="a"),
    T(f"{n10['n']}/{n9['n']} × {g10:.3f} + {n11['n']}/{n9['n']} × {g11:.3f} =", 900, 450, tam=44, c="tinta", w="sb", anc="rm", en="b"),
    N(gp9, 930, 450, "{:.3f}", 70, "gris", anc="lm", en="b", t=0.5),
    T(f"{g9:.3f} − {gp9:.3f} =", 900, 590, tam=60, c="tinta", w="sb", anc="rm", en="c"), N(gan9, 930, 590, "{:.3f}", 90, "rojo", anc="lm", en="c", t=0.4),
    P(f"{gan9 / gan_fallas_raiz:.0f} veces su ganancia en la raíz", 960, 760, "rojo", 48, en="c", t=1.0),
    T("útil en conjunto, como en la clase 33", 960, 870, tam=36, c="violeta", w="sb", en="c", t=1.4), real(1050))

# ---- 8. Gini o entropía
v.b("¿Y si en vez de Gini usas entropía? [[a]]El notebook de la clase dice que dan árboles casi idénticos, y en clase se dijo que no hay tanta diferencia. [[b]]Como regla general es cierto. [[c]]Pero con estos datos, un árbol de profundidad cinco da cero coma nueve cero seis con Gini, [[d]]y cero coma nueve dos cero con entropía. Uno coma cuatro puntos de AUC.",
    CL(), alerta(170, texto="matiz del material"),
    T("«dan árboles casi idénticos» (comentario del notebook)", 960, 300, tam=40, c="tinta", w="sb", en="a"),
    Bx(560, 840 - (D["crit"]["gini"] - 0.85) * 5000, 860, 840, "violeta", None, 0, 12, grow="y", en="c"),
    N(D["crit"]["gini"], 710, 840 - (D["crit"]["gini"] - 0.85) * 5000 - 44, "{:.4f}", 56, "violeta", en="c", t=0.3), T("Gini", 710, 880, tam=36, c="violeta", w="sb", en="c"),
    Bx(1060, 840 - (D["crit"]["entropy"] - 0.85) * 5000, 1360, 840, "azul", None, 0, 12, grow="y", en="d"),
    N(D["crit"]["entropy"], 1210, 840 - (D["crit"]["entropy"] - 0.85) * 5000 - 44, "{:.4f}", 56, "azul", en="d", t=0.3), T("entropía", 1210, 880, tam=36, c="azul", w="sb", en="d"),
    L(480, 840, 1440, 840, "tinta", 4), T("eje desde 0.85 · AUC con validación cruzada · profundidad 5", 960, 950, tam=28, c="gris", w="sb"),
    T("como regla general, cierto", 960, 380, tam=36, c="gris", w="sb", en="b"), sec="8. ¿Gini o entropía?")
v.b("La lección: [[a]]criterion es un hiperparámetro más. [[b]]No lo decidas por costumbre: pruébalo con validación cruzada, como la profundidad.",
    CL(), T("criterion = \"gini\" | \"entropy\"", 960, 340, tam=60, c="tinta", w="m", en="a"),
    P("es un hiperparámetro: decídelo con validación cruzada", 960, 540, "teal", 50, en="b"))

# ---- 9. Error típico
v.b("El error típico: [[a]]creer que el árbol que te entrega sklearn es el mejor árbol posible. [[b]]No lo es. Es el resultado de una cadena de decisiones voraces. [[c]]Un corte malo arriba condiciona todo lo que viene abajo.",
    CL(), T("«es el mejor árbol posible»", 960, 300, tam=64, c="tinta", en="a", id="e1"), U("e1", tachado=1.0, color="gris", en="b", dur=0.6),
    P("es una cadena de decisiones voraces", 960, 480, "violeta", 52, en="b", t=0.4), P("un corte malo arriba condiciona todo lo de abajo", 960, 640, "rojo", 48, en="c"),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Entre los que no reclamaron, segundos ganó por seis diezmilésimas. [[b]]Si reentrenas el árbol con otra muestra de clientes, ¿qué crees que pasa con ese segundo nivel?",
    CL(), P(f"segundos {N2N[0][2]:.4f} · frec_llamadas {N2N[1][2]:.4f} · clv {N2N[2][2]:.4f}", 960, 320, "ambar", 44, en="a"),
    T("¿y con otra muestra?", 960, 520, tam=70, c="tinta", en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Con un empate así, basta mover unos pocos clientes para que gane otra variable. [[a]]Y como el algoritmo es voraz, ese cambio arrastra todo lo que cuelga debajo. [[b]]Eso tiene nombre: inestabilidad. La vas a medir en el video nueve.",
    CL(), P("unos pocos clientes cambian el ganador", 960, 300, "ambar", 52, ent="subir"),
    T("y el cambio arrastra todo lo de abajo", 960, 460, tam=48, c="tinta", w="sb", en="a"), P("inestabilidad", 960, 640, "rojo", 66, en="b"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el árbol prueba todas las variables y todos los umbrales, y se queda con la mayor ganancia. [[b]]Lo repite en cada nodo, sin mirar adelante ni volver atrás. [[c]]Y una variable débil en la raíz puede ser fuerte más abajo.",
    CL(), P("todas las variables × todos los umbrales", 960, 300, "violeta", 52, en="a"),
    P("voraz: sin mirar adelante ni atrás", 960, 480, "rojo", 54, en="b"), P("débil arriba ≠ inútil abajo", 960, 660, "teal", 54, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]el gran peligro del árbol. El sobreajuste, y cómo medirlo con validación cruzada.",
    CL(), siguiente(6, "Sobreajuste y validación cruzada"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
