import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_06_varianza.json")
v = V("Varianza explicada y cuántos componentes")
SS, SR, CV, PR = D["scree_s"], D["real"], D["cv_auc"], D["prueba"]

v.portada("¿Cuántos componentes?", "Varianza explicada, scree plot y el criterio que de verdad decide",
          "Quinta parada: la varianza explicada, y la gran pregunta. ¿Con cuántos componentes me quedo?")

# ---- 1. Hook
v.b("Imagina que vas de viaje con una maleta chica. [[a]]Lo primero que guardas es lo más importante: el pasaporte, la ropa. [[b]]Cada cosa extra aporta menos. [[c]]En algún momento, agregar otra cosa ya no vale el espacio.",
    CL(), P("una maleta chica", 960, 220, "violeta", 56, ent="subir"),
    *[Bx(420 + 150 * i, 820 - h, 540 + 150 * i, 820, c, None, 0, 10, grow="y", en=k, t=0.1 * i)
      for i, (h, c, k) in enumerate(((420, "violeta", "a"), (260, "violeta", "a"), (160, "azul", "b"), (110, "azul", "b"), (70, "gris", "c"), (45, "gris", "c"), (30, "gris", "c")))],
    L(380, 820, 1500, 820, "tinta", 4), P("ya no vale el espacio", 1500, 560, "gris", 44, en="c", t=0.5), ilustrativo(990, texto="analogía"), sec="1. La maleta")
v.b("Con los componentes pasa igual. [[a]]PC1 guarda la mayor parte de la varianza, PC2 un poco menos, y así. [[b]]La pregunta es dónde parar.",
    CL(), P("PC1 guarda más · PC2 menos · …", 960, 380, "violeta", 56, en="a"), T("¿dónde parar?", 960, 600, tam=90, c="tinta", en="b"))

# ---- 2. Fórmula
v.b("Primero, cómo se mide cuánto guarda cada uno. [[a]]La varianza explicada del componente j es su lambda, dividido en la suma de todos los lambda. [[b]]Y con p variables estandarizadas, la suma de los lambda es justamente p.",
    CL(), T("varianza explicada", 960, 200, tam=56, c="azul", ent="subir"), caja_formula("λⱼ / Σλ", y=400, c="azul", tam=80, w=800, en="a"),
    P("con p variables estandarizadas: Σλ = p", 960, 640, "teal", 52, en="b"), T("en sklearn: explained_variance_ratio_", 960, 790, tam=38, c="tinta", w="m", en="b", t=0.6),
    sec="2. La fórmula")

# ---- 3. Ejemplo a mano
LAM = [2.4, 1.5, 1.0, 0.6, 0.3, 0.2]
VE = [40.0, 25.0, 16.7, 10.0, 5.0, 3.3]
AC = [40.0, 65.0, 81.7, 91.7, 96.7, 100.0]
v.b("Un ejemplo del apunte: seis variables estandarizadas. [[a]]Los valores propios son dos coma cuatro, uno coma cinco, uno, cero coma seis, cero coma tres y cero coma dos. [[b]]Suman seis: igual al número de variables.",
    CL(), T("ejemplo del apunte · 6 variables estandarizadas", 960, 180, tam=38, c="gris", w="sb"),
    *[[Bc(330 + 252 * i, 330, 210, 120, "blanco", "violeta", 4, 22, en="a", t=0.1 * i, ent="pop"), T(f"λ{i + 1} = {l}", 330 + 252 * i, 330, tam=40, c="violeta", en="a", t=0.1 * i + 0.05)] for i, l in enumerate(LAM)],
    T("2.4 + 1.5 + 1.0 + 0.6 + 0.3 + 0.2 = 6.0", 960, 500, tam=52, c="tinta", en="b"), K(1600, 500, "check", 60, en="b", t=0.8), sec="3. Un ejemplo a mano")
v.b("Ahora dividimos cada uno por seis. [[a]]Dos coma cuatro entre seis: cuarenta por ciento. [[b]]Uno coma cinco entre seis: veinticinco. [[c]]Uno entre seis: dieciséis coma siete. Y cero coma seis entre seis: diez por ciento.",
    *[[T(f"{l} / 6", 330 + 252 * i, 620, tam=36, c="tinta", w="sb", en=k), N(ve, 330 + 252 * i, 700, "{:.1f}", 50, "azul", suf=" %", en=k, t=0.2, dur=0.6)]
      for i, (l, ve, k) in enumerate(zip(LAM, VE, ["a", "b", "c", "c", "c", "c"]))])
v.b("Y vamos acumulando. [[a]]Cuarenta. Cuarenta más veinticinco, sesenta y cinco. [[b]]Más dieciséis coma siete, ochenta y uno coma siete. [[c]]Más diez, noventa y uno coma siete.",
    T("acumulada", 960, 764, tam=30, c="verde", w="sb"),
    *[[N(ac, 330 + 252 * i, 820, "{:.1f}", 50, "verde", suf=" %", en=k, t=0.15 * (i % 2), dur=0.6)] for i, (ac, k) in enumerate(zip(AC, ["a", "a", "b", "c", "c", "c"]))])
v.b("¿Cuántos componentes para llegar al ochenta por ciento? [[a]]Tres: es el primero que lo supera, con ochenta y uno coma siete. [[b]]¿Y para el noventa? Cuatro, con noventa y uno coma siete.",
    R(330 + 252 * 2, 820, 230, 90, "verde", 30, id="r80", en="a", ent="fade"), P("80 % → k = 3", 460, 960, "verde", 44, en="a", t=0.3),
    R(330 + 252 * 3, 820, 230, 90, "azul", 30, id="r90", en="b", ent="fade"), P("90 % → k = 4", 960, 960, "azul", 44, en="b", t=0.3))
v.b("Hay un tercer criterio, el de Kaiser: [[a]]quedarse con los componentes cuyo lambda es mayor que uno. [[b]]Aquí, solo dos coma cuatro y uno coma cinco. El tercero vale justo uno, y no es mayor. Kaiser dice k igual a dos.",
    X("r80", "r90", dur=0.3), R(330, 330, 230, 140, "rojo", 30, id="k1", en="b", ent="fade"), R(582, 330, 230, 140, "rojo", 30, id="k2", en="b", ent="fade", t=0.2),
    P("Kaiser (λ > 1) → k = 2", 1500, 960, "rojo", 44, en="b", t=0.6))
v.b("Tres criterios, tres respuestas: tres, cuatro y dos. [[a]]Y está bien. La profesora insistió: son referencias, no verdades. Hay que bajarlas al contexto.",
    CL(), *[[Bc(x, 460, 440, 280, "blanco", c, 5, 30, sombra=True, ent="subir", t=t_), T(n, x, 400, tam=44, c=c, t=t_ + 0.1), T(k_, x, 500, tam=70, c="tinta", t=t_ + 0.2)]
            for x, c, n, k_, t_ in ((460, "verde", "80 %", "k = 3", 0.0), (960, "azul", "90 %", "k = 4", 0.2), (1460, "rojo", "Kaiser", "k = 2", 0.4))],
    P("son referencias, no verdades", 960, 780, "violeta", 54, en="a"))

# ---- 4. Scree plot real
GS = dict(x0=260, y0=220, x1=1300, y1=860)
v.b("Ahora, los datos de la clase: quince columnas estandarizadas. [[a]]Este es el scree plot: una barra por componente, con su varianza explicada. [[b]]PC1 tiene lambda cinco coma veinticuatro: cinco coma veinticuatro entre quince, treinta y cuatro coma nueve por ciento.",
    CL(),
    G(GS["x0"], GS["y0"], GS["x1"], GS["y1"], 0.4, 15.6, 0, 100, [dict(xs=list(range(1, 16)), ys=SS["acum"], color="ambar", grosor=6, marcadores=8, revela=0.0)],
      xticks=list(range(1, 16)), yticks=[0, 20, 40, 60, 80, 100], yfmt="{:.0f}", xlabel="componente", id="gs", t=0.0),
    T("% de varianza", 260, 170, tam=32, c="tinta", w="sb", anc="lm"),
    *[Bx(k - 0.34, SS["var"][k - 1], k + 0.34, 0, "violeta", None, 0, 6, ref="gs", grow="y", en="a", t=0.06 * k, id=f"sb{k}") for k in range(1, 16)],
    g("sd", T(f"λ₁ = {SS['eig'][0]:.2f}", 1600, 360, tam=56, c="violeta", en="b"), T("5.24 / 15 =", 1520, 470, tam=46, c="tinta", en="b", t=0.3),
      N(SS["var"][0], 1600, 560, "{:.1f}", 80, "violeta", suf=" %", en="b", t=0.5, dur=0.7)), sec="4. El scree plot de la clase")
v.b("Después de PC1, la caída es fuerte, [[a]]y luego las barras quedan casi planas, entre seis y nueve por ciento. Ese quiebre es el codo. [[b]]La línea naranja es la acumulada.",
    g("sd", P("el codo: después de PC1", 1600, 700, "ambar", 42, en="a")),
    U("gs", **{"series.0.revela": 1.0}, en="b", dur=2.2), g("sd", T("acumulada", 1600, 800, tam=36, c="ambar", w="sb", en="b")))
v.b("Con siete componentes, la acumulada llega a setenta y ocho coma siete. [[a]]Con ocho, a ochenta y cinco. [[b]]Y con nueve, a noventa coma cinco. Para el ochenta por ciento se necesitan ocho componentes.",
    XG("sd", dur=0.3), Bc(7, SS["acum"][6], 26, 26, "blanco", "rojo", 5, 13, ref="gs", ent="pop", id="m7"), T(f"7 → {SS['acum'][6]:.1f} %", 1600, 360, tam=44, c="rojo", id="t7", t=0.2),
    Bc(8, SS["acum"][7], 26, 26, "blanco", "verde", 5, 13, ref="gs", ent="pop", en="a"), T(f"8 → {SS['acum'][7]:.1f} %", 1600, 450, tam=44, c="verde", en="a", t=0.2),
    Bc(9, SS["acum"][8], 26, 26, "blanco", "azul", 5, 13, ref="gs", ent="pop", en="b"), T(f"9 → {SS['acum'][8]:.1f} %", 1600, 540, tam=44, c="azul", en="b", t=0.2),
    L(0.4, 80, 15.6, 80, "verde", 3, guion=True, ref="gs", en="b", t=0.6))
v.b("Y aquí hay una alerta del material. [[a]]La diapositiva dice «siete componentes, aproximadamente ochenta por ciento», pero su tabla usa ocho. [[b]]Con los datos del notebook, siete dan setenta y ocho coma siete: no llegan. [[c]]En la base real de tres mil ciento cincuenta clientes, siete sí llegan: ochenta coma cuatro.",
    CL(), P("alerta del material", 960, 200, "ambar", 48, ent="subir"),
    T("«7 componentes (≈ 80 %)»", 960, 350, tam=60, c="gris", id="q7", en="a"), U("q7", tachado=1.0, en="b", dur=0.6),
    P(f"datos del notebook: 7 → {SS['acum'][6]:.1f} %  ·  8 → {SS['acum'][7]:.1f} %", 960, 530, "rojo", 46, en="b", t=0.3),
    P(f"base real (3.150): 7 → {SR['acum'][6]:.1f} %", 960, 690, "verde", 46, en="c"), T(f"PC1 de la base real: {SR['var'][0]:.1f} %", 960, 820, tam=36, c="gris", w="sb", en="c", t=0.6),
    sec="5. Una alerta del material")

# ---- 6. El criterio que decide
GC = dict(x0=260, y0=240, x1=1300, y1=860)
v.b("Pero el mejor criterio no es un porcentaje. [[a]]Es el desempeño del modelo, medido con validación cruzada, dentro de un Pipeline: escalar, PCA con k componentes, y regresión logística. [[b]]Probamos k de uno a quince.",
    CL(), P("el criterio que decide: desempeño en validación cruzada", 960, 220, "violeta", 46, ent="subir"),
    *[[Bc(x, 480, 340, 160, "blanco", c, 5, 26, sombra=True, en="a", ent="subir", t=t_), T(n, x, 480, tam=40, c=c, en="a", t=t_ + 0.1)]
      for x, c, n, t_ in ((480, "teal", "StandardScaler", 0.0), (960, "azul", "PCA(k)", 0.25), (1440, "violeta", "logística", 0.5))],
    L(660, 480, 780, 480, "gris", 6, True, en="a", t=0.3), L(1140, 480, 1260, 480, "gris", 6, True, en="a", t=0.55),
    P("k = 1, 2, …, 15 · 5-Fold · AUC", 960, 700, "gris", 48, en="b"), sec="6. El criterio que decide")
v.b("Este es el resultado. [[a]]Con un componente, el AUC es cero coma cincuenta y cinco. [[b]]Con tres, salta a cero coma setecientos setenta y tres, el máximo. [[c]]Y después baja un poco: con ocho, cero coma setecientos cuarenta y cinco. Con las quince variables, cero coma setecientos cuarenta y seis.",
    CL(),
    G(GC["x0"], GC["y0"], GC["x1"], GC["y1"], 0.5, 15.5, 0.5, 0.8, [dict(xs=list(range(1, 16)), ys=CV["medias"], color="violeta", grosor=7, marcadores=10, revela=0.0)],
      xticks=list(range(1, 16)), yticks=[0.5, 0.6, 0.7, 0.8], yfmt="{:.1f}", xlabel="k (número de componentes)", id="gc", t=0.0),
    T("AUC medio · 5-Fold", 260, 190, tam=32, c="tinta", w="sb", anc="lm"),
    U("gc", **{"series.0.revela": 0.01}, en="a", dur=0.4), P(f"k = 1 → {CV['medias'][0]:.3f}", 1600, 360, "gris", 42, en="a"),
    U("gc", **{"series.0.revela": 2 / 14}, en="b", dur=0.8), P(f"k = 3 → {CV['medias'][2]:.3f}", 1600, 470, "violeta", 42, en="b", t=0.6),
    U("gc", **{"series.0.revela": 1.0}, en="c", dur=1.6), P(f"k = 8 → {CV['medias'][7]:.3f}", 1600, 580, "azul", 42, en="c", t=1.0),
    L(0.5, CV["vars15"], 15.5, CV["vars15"], "gris", 3, guion=True, ref="gc", en="c", t=1.6), P(f"15 variables → {CV['vars15']:.3f}", 1600, 690, "gris", 42, en="c", t=1.8))
v.b("¿Cómo es posible que tres componentes rindan mejor que todo? [[a]]Porque varianza no es relevancia. PC3, que es un componente chico, es el que más se relaciona con la fuga. [[b]]Los componentes cuatro a quince agregan, sobre todo, ruido.",
    P("varianza ≠ relevancia", 1600, 820, "rojo", 46, en="a"), T("PC3: el más ligado a la fuga (video 7)", 1600, 900, tam=30, c="gris", w="sb", en="a", t=0.5, mw=520))

# ---- 7. Prueba
v.b("Ahora la comparación final en la prueba, la tabla de la diapositiva. [[a]]Con las quince variables: F1 cero coma doscientos setenta, MCC cero coma doscientos sesenta y cinco. [[b]]Con ocho componentes: F1 cero coma doscientos sesenta y siete, MCC cero coma doscientos cincuenta y dos.",
    CL(), T("en la prueba · regresión logística · umbral 0.5", 960, 190, tam=38, c="gris", w="sb"),
    *[[Bc(960, y, 1300, 150, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, 400, y, tam=44, c=c, anc="lm", en=k, t=0.1),
       T(f"F1 {PR[key]['F1']:.3f}", 1080, y, tam=48, c="tinta", en=k, t=0.2), T(f"MCC {PR[key]['MCC']:.3f}", 1430, y, tam=48, c="tinta", en=k, t=0.3)]
      for y, c, n, key, k in ((380, "gris", "15 variables", "15", "a"), (580, "violeta", "8 componentes", "pca8", "b"))],
    sec="7. La prueba: ¿cuesta desempeño?")
v.b("La diapositiva concluye que se cede algo de desempeño. [[a]]Pero en la prueba hay solo cincuenta y siete fugas. [[b]]Con tan pocas, uno o dos aciertos cambian el F1 en el segundo decimal. Esa diferencia es ruido. Y en validación cruzada, como vimos, tres componentes incluso ganan.",
    P(f"solo {D['fugas_te']} fugas en la prueba", 960, 760, "ambar", 50, en="a"), P("0.270 frente a 0.267: ruido, no una diferencia", 960, 890, "rojo", 46, en="b"))

# ---- 8. Error típico
v.b("Un error típico, que se escuchó en la clase: [[a]]«ochenta y cinco por ciento: con ocho componentes cubrimos a los que se fugan». [[b]]Son dos cantidades sin relación. El ochenta y cinco por ciento de varianza no tiene nada que ver con el ochenta y cinco por ciento de clientes que no se fugan.",
    CL(), T("«con 8 componentes cubrimos el 85 % de la fuga»", 960, 330, tam=54, c="tinta", id="q85", en="a"), U("q85", tachado=1.0, en="b", dur=0.6),
    Bc(560, 620, 680, 220, "blanco", "violeta", 5, 30, sombra=True, en="b", ent="subir"), T("85 % de la varianza", 560, 590, tam=46, c="violeta", en="b", t=0.1), T("(8 componentes)", 560, 660, tam=34, c="gris", w="sb", en="b", t=0.2),
    T("≠", 960, 620, tam=110, c="rojo", en="b", t=0.4),
    Bc(1360, 620, 680, 220, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir", t=0.5), T("85 % no se fuga", 1360, 590, tam=46, c="verde", en="b", t=0.6), T("(la fuga es 15 %)", 1360, 660, tam=34, c="gris", w="sb", en="b", t=0.7),
    sec="8. El error típico")

# ---- 9. Piensa
v.b("Piensa un segundo: [[a]]cinco variables estandarizadas, con lambda dos coma cinco, uno coma dos, cero coma siete, cero coma cuatro y cero coma dos. ¿Cuántos componentes necesitas para el ochenta por ciento?",
    CL(), T("5 variables estandarizadas", 960, 260, tam=46, c="gris", w="sb", en="a"),
    T("λ = 2.5 · 1.2 · 0.7 · 0.4 · 0.2", 960, 400, tam=64, c="violeta", en="a", t=0.3), T("¿k para 80 %?", 960, 580, tam=80, c="tinta", en="a", t=0.8),
    ilustrativo(990, en="a"), sec="9. Piensa un segundo", pausa=0.8)
v.b("La suma es cinco. [[a]]Dos coma cinco entre cinco: cincuenta por ciento. [[b]]Más uno coma dos entre cinco, veinticuatro: setenta y cuatro. Todavía no. [[c]]Más cero coma siete entre cinco, catorce: ochenta y ocho. Tres componentes.",
    CL(), T("Σλ = 5", 960, 220, tam=56, c="gris", w="sb"),
    *[[Bc(x, 480, 400, 280, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(op, x, 420, tam=40, c="tinta", w="sb", en=k, t=0.1), N(ac, x, 520, "{:.0f}", 80, c, suf=" %", en=k, t=0.3, dur=0.6)]
      for x, c, op, ac, k in ((460, "gris", "2.5 / 5", 50, "a"), (960, "gris", "+ 1.2 / 5", 74, "b"), (1460, "verde", "+ 0.7 / 5", 88, "c"))],
    P("k = 3", 960, 760, "verde", 70, en="c", t=0.8))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]la varianza explicada es lambda, dividido en la suma de los lambda. [[b]]Umbral, codo y Kaiser son referencias. [[c]]Y el número final de componentes se elige con validación cruzada, dentro de un Pipeline.",
    CL(), P("varianza explicada = λⱼ / Σλ", 960, 300, "azul", 54, en="a"), P("umbral · codo · Kaiser = referencias", 960, 480, "ambar", 50, en="b"),
    P("se decide con validación cruzada en un Pipeline", 960, 660, "violeta", 50, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]leer qué hay dentro de cada componente, y qué pasa si no estandarizas.",
    CL(), siguiente(7, "Leer las cargas y el efecto de la escala"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
