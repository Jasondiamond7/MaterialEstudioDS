import math
import random
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, codigo

OUT = out("guion_c34_09_limites.json")
v = V("Fortalezas y límites del árbol")
MO, IN, SR = D["mono"], D["inest"], D["sin_reclamos"]
UMB = {u[0]: u for u in D["mono_umbrales"]}
CU = D["curva"]
best = max(CU, key=lambda c: c["va"])
LOG = D["final"]["logistica"]

v.portada("Fortalezas y límites", "Lo que no le importa al árbol, lo que sí le duele, y cuándo conviene",
          "Novena parada. El árbol tiene virtudes que ningún otro modelo de la carpeta tiene, y defectos que hay que conocer. Hoy las comprobamos con números.")

# ---- 1. Invariancia
v.b("En la clase treinta y dos insistimos en escalar las variables antes de modelar. [[a]]Con un árbol, eso no hace falta. [[b]]Y no es una opinión: lo vamos a comprobar con un número.",
    CL(), T("clase 32: «escala antes de modelar»", 960, 280, tam=56, c="gris", ent="subir"),
    P("con un árbol, no hace falta", 960, 460, "verde", 60, en="a"), P("lo comprobamos con un número", 960, 620, "violeta", 48, en="b"),
    sec="1. No hay que escalar")
v.b("La razón: el árbol solo pregunta si x es menor o igual que un umbral. [[a]]Si aplicas una transformación que respeta el orden, como un logaritmo o una raíz, [[b]]x menor o igual que u ocurre exactamente cuando g de x es menor o igual que g de u. [[c]]El umbral cambia de número, pero los clientes que quedan a cada lado son los mismos.",
    CL(), caja_formula("x ≤ u   si y solo si   g(x) ≤ g(u)", y=330, c="violeta", tam=64, en="b", w=1300),
    P("g respeta el orden: log, raíz, cuadrado de positivos…", 960, 520, "teal", 44, en="a"),
    P("cambia el número del umbral, no la partición", 960, 680, "verde", 50, en="c"))

# ---- 2. Ejemplo resuelto
v.b("El experimento del notebook. [[a]]Aplica logaritmo a segundos, y raíz cuadrada a ce ele ve. [[b]]El umbral de segundos, novecientos treinta y cuatro, pasa a ser el logaritmo de uno más novecientos treinta y cuatro: seis coma ocho cuatro. [[c]]Y el de ce ele ve, sesenta y uno coma tres cinco, pasa a su raíz: siete coma ocho tres.",
    CL(), T("mismo árbol, variables transformadas", 960, 180, tam=48, c="tinta", ent="subir"),
    T("segundos → log(1 + segundos)      clv → √clv", 960, 300, tam=44, c="gris", w="sb", en="a"),
    T(f"segundos ≤ {UMB['segundos'][1]:g}   →   log(1 + 934) = {math.log1p(934):.3f}", 960, 460, tam=46, c="violeta", w="sb", en="b"),
    T(f"clv ≤ {UMB['clv'][1]:.2f}   →   √61.35 = {math.sqrt(61.35):.3f}", 960, 580, tam=46, c="violeta", w="sb", en="c"),
    T(f"umbral aprendido con las variables transformadas: {UMB['segundos'][2]:.3f} y {UMB['clv'][2]:.3f}", 960, 700, tam=34, c="gris", w="sb", en="c", t=0.8),
    real(1050), sec="2. Ejemplo: el mismo árbol")
v.b("¿Cambió quién cae en cada hoja? [[a]]No: los tres mil ciento cincuenta clientes caen exactamente en la misma hoja que antes. [[b]]Y el AUC con validación cruzada de un árbol de profundidad cinco: cero coma nueve cero cinco ocho antes, y cero coma nueve cero cinco ocho después. [[c]]Diferencia: exactamente cero. La logística, en cambio, pasa de cero coma nueve uno nueve nueve a cero coma nueve dos tres uno.",
    CL(), P(f"los {mil(D['n'])} clientes, en la misma hoja", 960, 190, "verde", 48, en="a") if D["mono_hojas_iguales"] else None,
    T("original", 900, 330, tam=34, c="gris", w="sb"), T("log y raíz", 1220, 330, tam=34, c="gris", w="sb"), T("diferencia", 1540, 330, tam=34, c="gris", w="sb"),
    Bc(960, 440, 1500, 120, "blanco", "verde", 4, 22, sombra=True, en="b", ent="subir"), T("árbol (profundidad 5)", 300, 440, tam=38, c="tinta", w="sb", anc="lm", en="b", t=0.1),
    T(f"{MO['arbol'][0]:.4f}", 900, 440, tam=44, c="verde", w="b", en="b", t=0.2), T(f"{MO['arbol'][1]:.4f}", 1220, 440, tam=44, c="verde", w="b", en="b", t=0.4),
    T("0", 1540, 440, tam=48, c="verde", w="b", en="c"),
    Bc(960, 600, 1500, 120, "blanco", "azul", 4, 22, sombra=True, en="c", ent="subir"), T("regresión logística", 300, 600, tam=38, c="tinta", w="sb", anc="lm", en="c", t=0.1),
    T(f"{MO['log'][0]:.4f}", 900, 600, tam=44, c="azul", w="b", en="c", t=0.3), T(f"{MO['log'][1]:.4f}", 1220, 600, tam=44, c="azul", w="b", en="c", t=0.5),
    T(f"+{MO['log'][1] - MO['log'][0]:.4f}", 1540, 600, tam=44, c="azul", w="b", en="c", t=0.7),
    P("con árboles, escalar no cambia nada: ni ayuda ni daña", 960, 800, "violeta", 46, en="c", t=1.2), real(1050))

# ---- 3. Inestabilidad
v.b("Ahora lo que sí le duele: la inestabilidad. [[a]]El notebook entrena seis árboles con seis muestras distintas del setenta por ciento de los clientes. [[b]]En las seis, la raíz es reclamos: su ventaja es enorme. [[c]]Pero mira lo que elige justo debajo, en la rama de los que no reclamaron.",
    CL(), T("6 muestras distintas del 70 % · árbol de profundidad 4", 960, 170, tam=44, c="tinta", ent="subir"),
    *[[Bc(330 + 252 * i, 400, 220, 140, "blanco", "rojo", 4, 20, sombra=True, en="a", t=0.15 * i, ent="pop"),
       T(f"semilla {r['sem']}", 330 + 252 * i, 360, tam=26, c="gris", w="sb", en="a", t=0.15 * i),
       T(r["raiz"], 330 + 252 * i, 415, tam=32, c="rojo", w="b", en="b", t=0.1 * i)] for i, r in enumerate(IN)],
    P("raíz: reclamos, las 6 veces", 960, 255, "rojo", 48, en="b", t=0.7), real(1050), sec="3. Lo que sí le duele: la inestabilidad")
COL = {"clv": "violeta", "segundos": "azul", "frec_llamadas": "ambar"}
v.b("Semilla cero: ce ele ve. [[a]]Semilla uno: segundos. Semilla dos: ce ele ve otra vez. [[b]]Semilla tres: frecuencia de llamadas. [[c]]Semillas cuatro y cinco: segundos, pero con umbrales distintos: novecientos treinta y cinco y medio, y quinientos setenta y cuatro.",
    *[[L(330 + 252 * i, 470, 330 + 252 * i, 640, "grisclaro", 5, en=m, t=0.1), Bc(330 + 252 * i, 720, 230, 150, "blanco", COL[r["izq"]], 4, 20, sombra=True, en=m, ent="pop", t=0.2),
       T(r["izq"], 330 + 252 * i, 700, tam=28, c=COL[r["izq"]], w="b", en=m, t=0.3), T(f"≤ {r['umbral']:g}", 330 + 252 * i, 750, tam=28, c="tinta", w="sb", en=m, t=0.4)]
      for i, (r, m) in enumerate(zip(IN, [None, "a", "a", "b", "c", "c"]))])
v.b("¿Por qué pasa? [[a]]Acuérdate del video cinco: en ese nodo, segundos, frecuencia de llamadas y ce ele ve estaban casi empatadas. [[b]]Con otra muestra gana otra, y todo lo que cuelga debajo cambia. [[c]]El profesor lo llevó al negocio: con reentrenamientos trimestrales, cada vez le explicas a la empresa un árbol distinto.",
    CL(), P(f"video 5: {D['nivel2_sin_reclamo'][0][0][0]} {D['nivel2_sin_reclamo'][0][0][2]:.4f} · frec_llamadas {D['nivel2_sin_reclamo'][0][1][2]:.4f} · clv {D['nivel2_sin_reclamo'][0][2][2]:.4f}", 960, 260, "ambar", 40, en="a"),
    P("otra muestra → otro ganador → otro árbol", 960, 430, "rojo", 52, en="b"),
    tarjetas([(560, "gris", "trimestre 1", "«el riesgo está en clv»", "c"), (1360, "gris", "trimestre 2", "«no, está en segundos»", "c")], y=680, w=640, h=240))

# ---- 4. Otros límites
rnd = random.Random(9)
PT = [(round(rnd.uniform(0.3, 9.7), 2), round(rnd.uniform(0.3, 9.7), 2)) for _ in range(60)]
arr = [(x, y) for x, y in PT if y > x]
aba = [(x, y) for x, y in PT if y <= x]
EX = [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]
EY = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]
v.b("Otros dos límites. [[a]]Primero, solo corta en ángulo recto. Si dos clases están separadas por una diagonal, [[b]]el árbol necesita una escalera de muchos cortes para aproximarla, y aun así queda dentada.",
    CL(), G(380, 220, 1100, 900, 0, 10, 0, 10,
            [dict(xs=[p[0] for p in arr] + [50], ys=[p[1] for p in arr] + [5], color="azul", linea=None, marcadores=12, revela=1.0),
             dict(xs=[p[0] for p in aba] + [50], ys=[p[1] for p in aba] + [5], color="ambar", linea=None, marcadores=12, revela=1.0),
             dict(xs=EX, ys=EY, color="violeta", grosor=7, revela=0.0)],
            xticks=[0, 5, 10], yticks=[0, 5, 10], yfmt="{:.0f}", id="gd", en="a"),
    L(0, 0, 10, 10, "tinta", 4, guion=True, ref="gd", en="a", t=0.4),
    P("la frontera real: una diagonal", 1500, 330, "gris", 38, en="a", t=0.6),
    U("gd", **{"series.2.revela": 1.0}, en="b", dur=2.0), P("el árbol: una escalera", 1500, 450, "violeta", 38, en="b", t=0.4),
    ilustrativo(1010), sec="4. Otros límites")
v.b("Segundo, predice una constante en cada región. [[a]]Por eso no extrapola: nunca predice algo fuera de lo que vio en el entrenamiento. [[b]]Y el límite más incómodo: un árbol solo suele predecir peor que una logística bien especificada. [[c]]Aquí, el mejor árbol da cero coma nueve uno uno, y la logística cero coma nueve dos cero.",
    CL(), tarjetas([(560, "ambar", "constante por región", "no extrapola fuera de lo visto", "a"), (1360, "rojo", "un árbol solo", "suele rendir menos que una logística", "b")], y=400, w=660, h=300),
    T(f"mejor árbol (prof. {best['d']}) {best['va']:.3f}   ·   logística {LOG:.3f}", 960, 700, tam=46, c="tinta", w="sb", en="c"), real(1050))

# ---- 5. Reclamos domina
v.b("Un último punto que el profesor recalcó: reclamos domina. [[a]]Sola, como un árbol de un solo corte, ya da un AUC de cero coma seis nueve nueve. [[b]]Y si la sacas, el árbol de profundidad cinco cae de cero coma nueve cero seis a cero coma ocho seis cuatro. [[c]]Su pregunta fue directa: ¿qué pasa si mañana no tengo la variable de reclamos? ¿Se sostiene el modelo en el tiempo?",
    CL(), T("reclamos domina", 960, 180, tam=56, c="rojo", ent="subir"),
    Bc(560, 430, 640, 260, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("reclamos sola", 560, 350, tam=38, c="gris", w="sb", en="a", t=0.1),
    N(D["auc_reclamos_sola"], 560, 450, "{:.3f}", 84, "rojo", en="a", t=0.3),
    Bc(1360, 430, 640, 260, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir"), T("árbol prof. 5 sin reclamos", 1360, 350, tam=38, c="gris", w="sb", en="b", t=0.1),
    T(f"{MO['arbol'][0]:.3f} → {SR['arbol5']:.3f}", 1360, 450, tam=64, c="ambar", w="b", en="b", t=0.3),
    P("«¿y si mañana no tengo reclamos?»", 960, 720, "violeta", 50, en="c"), real(1050), sec="5. Cuando una variable explica casi todo")
v.b("Y dio un consejo que vale para cualquier modelo. [[a]]Cuando una sola variable explica casi todo, no aplaudas. [[b]]Quizás la población no es homogénea. [[c]]Considera separar el problema en submodelos por segmento, por ejemplo uno para los que reclamaron y otro para los que no. Estratifica.",
    CL(), T("una variable explica casi todo", 960, 230, tam=56, c="tinta", ent="subir"),
    P("no aplaudas", 960, 380, "rojo", 56, en="a"), P("quizás la población no es homogénea", 960, 520, "ambar", 48, en="b"),
    tarjetas([(560, "rojo", "submodelo", "los que reclamaron", "c"), (1360, "verde", "submodelo", "los que no reclamaron", "c")], y=760, w=600, h=220))

# ---- 6. Fortalezas
v.b("Junta todo en una lista de fortalezas. [[a]]Se dibuja y se explica a alguien que no sabe estadística. [[b]]Captura no linealidad e interacciones sin que nadie las escriba. [[c]]No hay que escalar. [[d]]Mezcla variables continuas y discretas. [[e]]Y selecciona variables mientras se entrena.",
    CL(), T("fortalezas", 960, 170, tam=52, c="verde", ent="subir"),
    *[[Bc(960, 290 + 120 * i, 1300, 96, "blanco", "verde", 3, 20, en=m, ent="subir"), K(380, 290 + 120 * i, "check", 40, en=m, t=0.1),
       T(t_, 440, 290 + 120 * i, tam=38, c="tinta", w="sb", anc="lm", en=m, t=0.15)]
      for i, (t_, m) in enumerate(zip(["se dibuja y se explica a cualquiera", "no linealidad e interacciones solas", "no hay que escalar (invariante a log, raíz…)",
                                        "continuas y discretas juntas", "selección de variables propia (embedded)"], "abcde"))],
    sec="6. Fortalezas")

# ---- 7. Error típico
v.b("El error típico: [[a]]escalar o transformar las variables para un árbol, esperando que mejore. [[b]]Acabas de ver que no cambia nada. Ese tiempo úsalo en lo que sí importa: la poda y la validación.",
    CL(), T("StandardScaler antes del árbol", 960, 300, tam=60, c="tinta", w="sb", en="a", id="e1"), U("e1", tachado=1.0, color="gris", en="b", dur=0.6),
    P("no cambia nada: el tiempo, para podar y validar", 960, 500, "verde", 50, en="b", t=0.4), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si a la antigüedad, que nunca es negativa, la elevas al cuadrado, ¿cambia el árbol? [[b]]¿Y si a segundos le restas su promedio y divides por su desvío?",
    CL(), P("antigüedad² (sin negativos)", 960, 300, "violeta", 54, en="a"), P("(segundos − media) / desvío", 960, 450, "violeta", 54, en="b"),
    T("¿cambia el árbol?", 960, 640, tam=70, c="tinta", en="b", t=0.5), sec="8. Piensa un segundo", pausa=0.6)
v.b("En los dos casos, no. [[a]]Elevar al cuadrado números positivos respeta el orden. [[b]]Y estandarizar es restar y dividir por un número positivo: también respeta el orden. [[c]]Mismas particiones, mismo árbol. Solo cambian los números de los umbrales.",
    CL(), T("en los dos casos: no", 960, 230, tam=64, c="verde", ent="subir"),
    T("x² con x ≥ 0 respeta el orden", 960, 400, tam=48, c="tinta", w="sb", en="a"), T("(x − media)/desvío: también", 960, 510, tam=48, c="tinta", w="sb", en="b"),
    P("mismas particiones · solo cambian los umbrales", 960, 690, "teal", 50, en="c"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]el árbol es invariante a transformaciones que respetan el orden: no hay que escalar. [[b]]Es inestable: la raíz se repite, pero los niveles de abajo cambian con la muestra. [[c]]Y solo, suele predecir peor que una logística. Se elige cuando la explicación vale más que el punto de AUC.",
    CL(), P("invariante a log, raíz…: no escalar", 960, 300, "verde", 52, en="a"),
    P("inestable: cambia con la muestra", 960, 480, "rojo", 52, en="b"), P("se elige por explicable, no por el AUC", 960, 660, "violeta", 52, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]si el problema es la inestabilidad, la solución es promediar muchos árboles. El salto al bosque.",
    CL(), siguiente(10, "El salto al bosque"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
