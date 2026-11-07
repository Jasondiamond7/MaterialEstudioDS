import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula

OUT = out("guion_c34_03_impureza.json")
v = V("Impureza: Gini y entropía")
pq, pf = D["n_queda"] / D["n"], D["n_fuga"] / D["n"]
RC = D["reclamos"]


def gini(p):
    return 1 - p * p - (1 - p) ** 2


def ent(p):
    return 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


v.portada("Impureza: Gini y entropía", "Cómo mide un árbol qué tan mezclado está un grupo",
          "Tercera parada. Antes de entender por qué el árbol eligió reclamos, necesitamos una regla para medir el desorden de un grupo. Hoy calculas Gini y entropía a mano.")

# ---- 1. Hook
v.b("En el video anterior, la raíz preguntó por reclamos. [[a]]¿Por qué reclamos, y no la edad, o los segundos? [[b]]El árbol no elige por intuición. [[c]]Prueba cortes y se queda con el que deja los grupos menos mezclados. Para eso necesita medir la mezcla.",
    CL(), T("¿por qué reclamos y no otra?", 960, 280, tam=66, c="tinta", en="a"),
    P("no elige por intuición", 960, 450, "gris", 48, en="b"), P("busca grupos menos mezclados", 960, 590, "violeta", 54, en="c"),
    sec="1. ¿Cómo mide el árbol?")

# ---- 2. Analogía: frascos
FR = [(420, "puro", 10, 0, "verde"), (960, "80 / 20", 8, 2, "ambar"), (1500, "50 / 50", 5, 5, "rojo")]
v.b("Piensa en tres frascos con pelotas. [[a]]El primero tiene solo pelotas verdes: es puro. [[b]]El segundo tiene ocho verdes y dos rojas. [[c]]El tercero, cinco y cinco: la mezcla máxima.",
    CL(), *[[Bc(x, 520, 400, 420, "blanco", c, 5, 40, sombra=True, en=k, ent="subir"), T(lab, x, 360, tam=44, c=c, en=k, t=0.1),
             Pt([(nv, [x - 140, 440, x + 140, 560 if nr else 660], "verde")] + ([(nr, [x - 140, 600, x + 140, 690], "rojo")] if nr else []), 22, en=k, t=0.2)]
           for (x, lab, nv, nr, c), k in zip(FR, "abc")],
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: tres frascos")
v.b("Ahora un juego. [[a]]Sacas una pelota al azar, la devuelves, y sacas otra. [[b]]¿Cuál es la probabilidad de que las dos sean de distinto color? [[c]]En el frasco puro, cero. En el de cincuenta y cincuenta, la mitad de las veces. [[d]]Esa probabilidad tiene nombre: es el índice de Gini.",
    T("saco 2 pelotas (devolviendo la primera)", 960, 150, tam=40, c="tinta", w="sb", en="a"),
    T("¿de distinto color?", 960, 830, tam=48, c="violeta", en="b"),
    N(0.0, 420, 900, "{:.2f}", 60, "verde", en="c"), N(0.5, 1500, 900, "{:.2f}", 60, "rojo", en="c", t=0.6),
    N(gini(0.2), 960, 900, "{:.2f}", 60, "ambar", en="d", t=0.4), P("= índice de Gini", 960, 990, "violeta", 40, en="d", t=0.8))

# ---- 3. Paso a paso: la fórmula
v.b("Paso a paso. [[a]]Sea pe sub i la proporción de la clase i dentro del nodo. [[b]]La probabilidad de sacar dos de la misma clase es la suma de los pe al cuadrado. [[c]]Y Gini es uno menos eso.",
    CL(), T("p_i = proporción de la clase i DENTRO del nodo", 960, 230, tam=46, c="tinta", w="sb", en="a"),
    T("misma clase dos veces:  p₀² + p₁²", 960, 400, tam=54, c="gris", en="b"),
    caja_formula("Gini = 1 − Σ pᵢ²", y=600, c="violeta", tam=80, en="c", w=900), sec="3. Paso a paso: las fórmulas")
v.b("Con dos clases, queda uno menos pe al cuadrado, menos uno menos pe, al cuadrado. [[a]]La otra medida es la entropía, que viene de la teoría de la información: la medida del desorden. [[b]]Es menos la suma de pe por logaritmo en base dos de pe.",
    CL(), caja_formula("Gini = 1 − p² − (1 − p)²", y=280, c="violeta", tam=62, w=1100),
    caja_formula("Entropía = − Σ pᵢ · log₂ pᵢ", y=560, c="azul", tam=62, en="a", w=1100),
    T("teoría de la información · «medida del desorden»", 960, 700, tam=36, c="azul", w="sb", en="a", t=0.5),
    T("log₂: logaritmo en base dos", 960, 800, tam=34, c="gris", w="sb", en="b"))

# ---- 4. Ejemplo resuelto: la raíz
v.b("Ejemplo con la raíz del árbol de la clase. [[a]]Hay tres mil ciento cincuenta clientes, y se fugan cuatrocientos noventa y cinco. [[b]]La proporción que se queda es dos mil seiscientos cincuenta y cinco sobre tres mil ciento cincuenta: cero coma ocho cuatro dos nueve. [[c]]La que se fuga: cero coma uno cinco siete uno.",
    CL(), T("la raíz · datos reales de la clase", 960, 170, tam=44, c="tinta", ent="subir"),
    Pt([(84, [300, 260, 1100, 520], "#94A3B8"), (16, [1160, 260, 1620, 520], "rojo")], 14, en="a"),
    T(f"p_queda = {mil(D['n_queda'])} / {mil(D['n'])} =", 940, 640, tam=48, c="tinta", w="sb", anc="rm", en="b"),
    N(pq, 970, 640, "{:.4f}", 60, "gris", anc="lm", en="b", t=0.5),
    T(f"p_fuga = {D['n_fuga']} / {mil(D['n'])} =", 940, 750, tam=48, c="tinta", w="sb", anc="rm", en="c"),
    N(pf, 970, 750, "{:.4f}", 60, "rojo", anc="lm", en="c", t=0.5), T("cada punto ≈ 1 % de los clientes", 960, 560, tam=28, c="gris", w="sb", en="a", t=0.5),
    real(1050), sec="4. Ejemplo: el Gini de la raíz")
v.b("Ahora Gini. [[a]]Cero coma ocho cuatro dos nueve al cuadrado da cero coma siete uno cero cuatro. [[b]]Cero coma uno cinco siete uno al cuadrado, cero coma cero dos cuatro siete. [[c]]Uno menos las dos: cero coma dos seis cuatro nueve. El Gini de la raíz es cero coma doscientos sesenta y cinco.",
    CL(), T("Gini de la raíz", 960, 180, tam=48, c="violeta", ent="subir"),
    T(f"{pq:.4f}² = {pq * pq:.4f}", 960, 340, tam=56, c="gris", w="sb", en="a"),
    T(f"{pf:.4f}² = {pf * pf:.4f}", 960, 450, tam=56, c="rojo", w="sb", en="b"),
    T(f"1 − {pq * pq:.4f} − {pf * pf:.4f} =", 900, 600, tam=56, c="tinta", w="sb", anc="rm", en="c"),
    N(D["gini_raiz"], 930, 600, "{:.4f}", 84, "violeta", anc="lm", en="c", t=0.5), real(1050))
v.b("Y la entropía de la misma raíz. [[a]]Cero coma ocho cuatro dos nueve por menos cero coma dos cuatro seis seis, en valor absoluto, da cero coma dos cero siete nueve. [[b]]Cero coma uno cinco siete uno por dos coma seis siete: cero coma cuatro uno nueve cinco. [[c]]Sumadas: cero coma seis dos siete cuatro.",
    CL(), T("entropía de la raíz", 960, 180, tam=48, c="azul", ent="subir"),
    T(f"log₂ {pq:.4f} = {math.log2(pq):.4f}     log₂ {pf:.4f} = {math.log2(pf):.3f}", 960, 290, tam=40, c="gris", w="sb", ent="subir"),
    T(f"{pq:.4f} × {-math.log2(pq):.4f} = {-pq * math.log2(pq):.4f}", 960, 420, tam=54, c="gris", w="sb", en="a"),
    T(f"{pf:.4f} × {-math.log2(pf):.3f} = {-pf * math.log2(pf):.4f}", 960, 530, tam=54, c="rojo", w="sb", en="b"),
    T(f"{-pq * math.log2(pq):.4f} + {-pf * math.log2(pf):.4f} =", 900, 680, tam=56, c="tinta", w="sb", anc="rm", en="c"),
    N(D["ent_raiz"], 930, 680, "{:.4f}", 84, "azul", anc="lm", en="c", t=0.5), real(1050))

# ---- 5. Las dos curvas
ps = [i / 40 for i in range(41)]
GC = dict(x0=300, y0=230, x1=1200, y1=860)
v.b("¿Cómo se comportan las dos medidas? [[a]]En el eje horizontal, la proporción de fuga del nodo. [[b]]Gini sube desde cero, llega a cero coma cinco con mitad y mitad, y vuelve a cero. [[c]]La entropía hace lo mismo, pero su máximo es uno.",
    CL(), G(GC["x0"], GC["y0"], GC["x1"], GC["y1"], 0, 1, 0, 1.05,
            [dict(xs=ps, ys=[gini(p) for p in ps], color="violeta", grosor=7, revela=0.0, suave=True),
             dict(xs=ps, ys=[ent(p) for p in ps], color="azul", grosor=7, revela=0.0, suave=True)],
            xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1.0], yfmt="{:.2f}", xlabel="proporción de fuga en el nodo (p)", id="gc", en="a"),
    U("gc", **{"series.0.revela": 1.0}, en="b", dur=2.0), P("Gini · máx 0.5", 1530, 420, "violeta", 44, en="b", t=0.6),
    U("gc", **{"series.1.revela": 1.0}, en="c", dur=2.0), P("entropía · máx 1", 1530, 300, "azul", 44, en="c", t=0.6),
    sec="5. Las dos curvas")
v.b("Las dos valen cero cuando el nodo es puro, y son máximas con la mezcla perfecta. [[a]]Nuestra raíz está aquí, en quince coma siete por ciento de fuga: [[b]]Gini cero coma dos seis cinco, entropía cero coma seis dos siete. [[c]]Gini se calcula más rápido, porque no usa logaritmos, y es el criterio por defecto en sklearn.",
    L(pf, 0, pf, 1.0, "rojo", 4, guion=True, ref="gc", en="a"), Bc(pf, gini(pf), 22, 22, "violeta", None, 0, 11, ref="gc", en="a", t=0.3),
    Bc(pf, ent(pf), 22, 22, "azul", None, 0, 11, ref="gc", en="a", t=0.4),
    T(f"raíz: p = {pf:.3f}", 1530, 560, tam=40, c="rojo", w="b", en="a", t=0.3),
    T(f"Gini {D['gini_raiz']:.3f} · entropía {D['ent_raiz']:.3f}", 1530, 640, tam=36, c="tinta", w="sb", en="b"),
    P("Gini: sin logaritmos · default de sklearn", 1530, 780, "teal", 32, en="c"))

# ---- 6. Tabla y alerta
FILAS = [("100 % se queda (puro)", "100/0", "verde"), ("84 % / 16 %  ← «nuestra raíz»", "84/16", "gris"),
         ("50 % / 50 % (máxima mezcla)", "50/50", "rojo"), ("6 % / 94 %  ← hoja de fuga", "6/94", "ambar")]
v.b("La diapositiva once resume cuatro nodos típicos. [[a]]Puro: cero y cero. [[b]]Ochenta y cuatro contra dieciséis. [[c]]Mitad y mitad: cero coma cinco y uno. [[d]]Y seis contra noventa y cuatro, como la hoja de fuga del árbol.",
    CL(), T("composición del nodo", 380, 230, tam=32, c="gris", w="sb", anc="lm"), T("Gini", 1230, 230, tam=32, c="violeta", w="sb"),
    T("entropía", 1530, 230, tam=32, c="azul", w="sb"),
    *[[Bc(960, 330 + 120 * i, 1300, 100, "blanco", c, 3, 18, en=k, ent="subir"), T(lab, 380, 330 + 120 * i, tam=38, c="tinta", w="sb", anc="lm", en=k, t=0.05),
       T(f"{D['tabla_imp'][key][0]:.3f}", 1230, 330 + 120 * i, tam=44, c="violeta", w="b", en=k, t=0.1),
       T(f"{D['tabla_imp'][key][1]:.3f}", 1530, 330 + 120 * i, tam=44, c="azul", w="b", en=k, t=0.15)]
      for i, ((lab, key, c), k) in enumerate(zip(FILAS, "abcd"))], sec="6. La tabla de la diapositiva 11")
v.b("Ahora, una alerta del material. [[a]]Esa fila dice que el Gini de nuestra raíz es cero coma doscientos sesenta y nueve. [[b]]Pero ochenta y cuatro contra dieciséis es un redondeo. [[c]]La raíz real tiene ochenta y cuatro coma tres contra quince coma siete, y su Gini exacto es cero coma doscientos sesenta y cinco, el mismo de la diapositiva anterior.",
    alerta(140), R(1230, 450, 200, 110, "ambar", 20, en="a"),
    T("84 / 16 es un redondeo", 960, 870, tam=40, c="ambar", w="b", en="b"),
    T(f"real: {100 * pq:.1f} / {100 * pf:.1f}  →  Gini {D['gini_raiz']:.3f}", 960, 950, tam=40, c="verde", w="b", en="c"))

# ---- 7. Error típico
v.b("El error típico con la impureza: [[a]]usar las proporciones de toda la base en un nodo hijo. [[b]]Los pe se recalculan en cada nodo, con los clientes que cayeron ahí. [[c]]En el nodo de los que reclamaron, la fuga no es quince coma siete: es setenta y cinco coma cinco. Y su Gini es cero coma tres siete, no cero coma dos seis cinco.",
    CL(), T("usar la p de toda la base en un hijo", 960, 260, tam=58, c="tinta", en="a", id="e1"), U("e1", tachado=1.0, color="gris", en="b", dur=0.6),
    P("los pᵢ se recalculan en CADA nodo", 960, 430, "verde", 54, en="b", t=0.4),
    Bc(960, 680, 1200, 240, "blanco", "rojo", 5, 30, sombra=True, en="c", ent="subir"),
    T(f"nodo «reclamó»: {RC['n_si']} clientes, {RC['f_si']} se fugan", 960, 620, tam=40, c="tinta", w="sb", en="c", t=0.1),
    T(f"p = {RC['f_si'] / RC['n_si']:.3f}  →  Gini = {gini(RC['f_si'] / RC['n_si']):.3f}", 960, 710, tam=52, c="rojo", w="b", en="c", t=0.4), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un nodo tiene veinte observaciones: quince de la clase A y cinco de la clase B. [[b]]¿Cuál es su Gini? Y si te animas, ¿su entropía?",
    CL(), Pt([(15, [520, 300, 1180, 420], "azul"), (5, [1240, 300, 1400, 420], "ambar")], 24, en="a"),
    T("15 de A · 5 de B", 960, 520, tam=56, c="tinta", w="sb", en="a", t=0.3),
    T("¿Gini? ¿entropía?", 960, 680, tam=68, c="violeta", en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("La proporción de A es cero coma setenta y cinco. [[a]]Al cuadrado, cero coma cinco seis dos cinco. La de B, cero coma veinticinco, al cuadrado cero coma cero seis dos cinco. [[b]]Gini: uno menos cero coma seis dos cinco, igual a cero coma tres siete cinco. [[c]]Y la entropía: cero coma tres uno uno más cero coma cinco: cero coma ocho uno uno.",
    CL(), T("p_A = 15 / 20 = 0.75", 960, 230, tam=54, c="azul", w="sb", ent="subir"),
    T("0.75² + 0.25² = 0.5625 + 0.0625 = 0.625", 960, 360, tam=50, c="gris", w="sb", en="a"),
    T("Gini = 1 − 0.625 =", 900, 500, tam=60, c="tinta", w="sb", anc="rm", en="b"), N(gini(0.75), 930, 500, "{:.3f}", 84, "violeta", anc="lm", en="b", t=0.4),
    T("0.75 × 0.415 + 0.25 × 2 = 0.311 + 0.5 =", 900, 680, tam=46, c="tinta", w="sb", anc="rm", en="c"),
    N(ent(0.75), 930, 680, "{:.3f}", 72, "azul", anc="lm", en="c", t=0.5), T("ejercicio de la diapositiva 19 (Arbol de decision.pptx)", 960, 850, tam=28, c="gris", w="sb", en="c", t=0.9))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]la impureza mide qué tan mezclado está un nodo. Vale cero si es puro. [[b]]Gini es uno menos la suma de los pe al cuadrado, con máximo cero coma cinco. Entropía, con máximo uno. [[c]]Y los pe se recalculan en cada nodo.",
    CL(), P("impureza = mezcla · 0 si es puro", 960, 300, "violeta", 54, en="a"),
    P("Gini máx 0.5 · entropía máx 1", 960, 480, "azul", 54, en="b"), P("los pᵢ son del nodo", 960, 660, "teal", 54, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]usar la impureza para elegir el primer corte. Lo vas a calcular a mano, con reclamos contra edad.",
    CL(), siguiente(4, "El primer corte, a mano"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
