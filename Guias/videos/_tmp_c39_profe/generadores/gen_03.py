import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas)

OUT = out("guion_c39_03_wss.json")
v = V("Distancia, WSS y TSS = WSS + BSS")
KM = D["km"]

v.portada("Distancia y WSS", "Qué minimiza K-means y cómo se reparte la variabilidad",
          "Tercera parada: la cuenta que hay debajo de K-means. Distancia al centro, al cuadrado, sumada dentro de cada grupo.")

# ---- 1. Hook
v.b("Todo K-means se apoya en una sola pregunta, repetida miles de veces: [[a]]¿qué tan lejos está este punto de este centro? [[b]]Si entiendes cómo se mide esa distancia, y qué se suma con ella, entiendes qué busca el algoritmo. [[c]]Y vas a poder leer sus números: el WSS, el BSS y el TSS.",
    CL(), T("una sola pregunta, repetida", 960, 150, tam=56, c="tinta", ent="subir"),
    P("¿qué tan lejos está este punto de este centro?", 960, 380, "violeta", 52, en="a"),
    P("WSS · BSS · TSS", 960, 600, "teal", 56, en="c"), sec="1. Una sola pregunta")

# ---- 2. Distancia
GD = "gd3"
v.b("La diapositiva diez hace la cuenta con dos puntos ya estandarizados. [[a]]Una observación en uno, dos, y un centro en tres, uno. [[b]]La distancia al cuadrado suma las diferencias al cuadrado en cada variable: uno menos tres, al cuadrado, es cuatro; dos menos uno, al cuadrado, es uno. Total: cinco. [[c]]La distancia euclidiana es la raíz de cinco, aproximadamente 2.236.",
    CL(), T("distancia euclidiana: una cuenta · diapositiva 10", 960, 140, tam=48, c="tinta", ent="subir"),
    G(400, 240, 940, 840, 0, 4, 0, 3, [dict(xs=[1, 3], ys=[2, 1], color="violeta", grosor=5, revela=0.0, lapiz=False, marcadores=12)],
      xticks=[0, 1, 2, 3, 4], yticks=[0, 1, 2, 3], yfmt="{:.0f}", xlabel="variable 1", ylabel="var. 2", id=GD, tam_txt=26, en="a"),
    U(GD, **{"series.0.revela": 1.0}, en="a", t=0.4, dur=0.8),
    T("x = (1, 2)", 1, 2.25, tam=32, c="azul", w="b", ref=GD, en="a", t=0.4), T("c = (3, 1)", 3, 0.72, tam=32, c="rojo", w="b", ref=GD, en="a", t=0.4),
    T("(1 − 3)² + (2 − 1)² = 4 + 1 = 5", 1380, 400, tam=46, c="tinta", w="sb", en="b"),
    T("distancia = √5 ≈", 1340, 560, tam=50, c="tinta", w="sb", anc="rm", en="c"), N(2.236, 1370, 560, "{:.3f}", 70, "violeta", anc="lm", en="c", t=1.0),
    diapo(10, en="a"), sec="2. Distancia euclidiana")
v.b("Y un detalle clave: [[a]]K-means minimiza la suma de las distancias al cuadrado, no de las distancias. [[b]]El cuadrado aumenta el peso de los puntos alejados. Un punto a distancia uno aporta uno; a distancia tres, aporta nueve. [[c]]Por eso un vino muy extremo puede tirar de un centro con mucha fuerza.",
    P("K-means minimiza distancias AL CUADRADO", 1340, 720, "rojo", 40, en="a"),
    T("distancia 1 → aporta 1   ·   distancia 3 → aporta 9", 1340, 820, tam=36, c="tinta", w="sb", en="b"),
    T("los puntos lejanos pesan más", 1340, 900, tam=36, c="ambar", w="sb", en="c"))

# ---- 3. Analogía
v.b("Una analogía. [[a]]Un grupo de amigos vive en la misma calle y quiere elegir dónde poner un punto de encuentro. [[b]]Si el costo de caminar crece con el cuadrado de la distancia, el mejor lugar es exactamente el promedio de las casas. [[c]]K-means hace eso con cada grupo: pone su centro en el promedio de sus miembros.",
    CL(), T("el punto de encuentro", 960, 150, tam=56, c="tinta", ent="subir"),
    L(300, 450, 1620, 450, "gris", 6, en="a"),
    [x for i, xx in enumerate([420, 620, 1380]) for x in [Bc(xx, 450, 60, 60, "azulclaro", "azul", 4, 12, en="a", t=0.15 * i),
                                                         T("casa", xx, 520, tam=28, c="azul", w="sb", en="a", t=0.15 * i)]],
    K(806.7, 450, "cruz", 60, "rojo", en="b"), T("promedio", 806.7, 360, tam=34, c="rojo", w="b", en="b", t=0.3),
    P("el centro de cada grupo = el promedio de sus miembros", 960, 720, "violeta", 46, en="c"), analogia(1050), sec="3. Analogía: el punto de encuentro")

# ---- 4. El criterio
v.b("Ahora el criterio completo, de la diapositiva once. [[a]]Buscamos {k|, ka,} grupos y sus centros que hagan mínimo el WSS: para cada grupo, sumamos las distancias al cuadrado de sus puntos a su centro, y después sumamos sobre todos los grupos. [[b]]Para un grupo ya formado, el centro que minimiza ese costo es su media. [[c]]Por eso el método se llama K-means: k medias.",
    CL(), T("K-means: criterio de ajuste · diapositiva 11", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("WSS = Σ (grupos)  Σ (puntos del grupo)  ‖ z − centro del grupo ‖²", y=320, c="violeta", tam=46, w=1600, en="a"),
    caja_formula("centro del grupo = media de sus puntos", y=520, c="teal", tam=50, w=1200, en="b"),
    P("K-means = k medias", 960, 720, "azul", 52, en="c"), diapo(11, en="a"), sec="4. El criterio WSS")
v.b("¿Por qué la media? Hagamos la prueba con un grupo de tres valores: uno, dos y seis. [[a]]Su media es tres. Las distancias al cuadrado son cuatro, uno y nueve: suman 14. [[b]]Si pusiéramos el centro en dos, la mediana, sumarían uno, cero y dieciséis: 17. [[c]]Y en cuatro, nueve, cuatro y cuatro: también 17. La media gana.",
    CL(), T("¿por qué la media? grupo: 1, 2 y 6", 960, 150, tam=52, c="tinta", ent="subir"),
    T("centro 3 (media):   4 + 1 + 9 =", 1060, 330, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(14, 1090, 330, "{:.0f}", 70, "verde", anc="lm", en="a", t=1.2),
    T("centro 2 (mediana):   1 + 0 + 16 =", 1060, 470, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(17, 1090, 470, "{:.0f}", 70, "rojo", anc="lm", en="b", t=1.2),
    T("centro 4:   9 + 4 + 4 =", 1060, 610, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(17, 1090, 610, "{:.0f}", 70, "rojo", anc="lm", en="c", t=1.0),
    P("la media minimiza la suma de cuadrados", 960, 790, "teal", 46, en="c", t=2.5), ilustrativo(1050))
v.b("Una advertencia de la misma diapositiva: [[a]]un WSS menor significa grupos más compactos, bajo esta escala y este criterio. [[b]]No demuestra, por sí solo, que encontramos grupos reales ni útiles.",
    P("WSS menor = más compacto (con esta escala)", 960, 900, "ambar", 42, en="a"),
    T("no prueba que los grupos sean reales ni útiles", 960, 990, tam=36, c="rojo", w="sb", en="b"))

# ---- 5. La descomposición
v.b("El WSS tiene dos compañeros. [[a]]El TSS es la variabilidad total: las distancias al cuadrado de cada punto a la media global, sin grupos. [[b]]El BSS mide la separación: compara el centro de cada grupo con la media global, y lo pondera por el tamaño del grupo. [[c]]Y para estas medias y las mismas observaciones se cumple: TSS es igual a WSS más BSS.",
    CL(), T("WSS, BSS y variabilidad total · diapositiva 16", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(400, "rojo", "TSS", "puntos contra la media global", "a"), (960, "azul", "WSS", "puntos contra el centro de su grupo", "a"),
              (1520, "teal", "BSS", "centros contra la media global, por tamaño", "b")], y=430, w=480, h=300, tam_t=56, tam_d=34),
    caja_formula("TSS = WSS + BSS", y=730, c="violeta", tam=64, w=900, en="c"), diapo(16, en="a"), sec="5. TSS = WSS + BSS")
v.b("Dos lecturas importantes. [[a]]Como el TSS no depende de los grupos, bajar el WSS es exactamente lo mismo que subir el BSS: grupos más compactos son grupos más separados. [[b]]Y el BSS no es el promedio de las distancias entre todos los pares de centros: es cada centro contra la media global, multiplicado por cuántos puntos tiene.",
    CL(), P("TSS fijo → bajar WSS = subir BSS", 960, 330, "violeta", 52, en="a"),
    K(330, 560, "cruz", 60, en="b"), T("BSS = promedio de distancias entre centros", 410, 560, tam=42, c="rojo", anc="lm", en="b"),
    K(330, 690, "check", 60, en="b", t=2.5), T("BSS = Σ tamaño × (centro − media global)²", 410, 690, tam=42, c="verde", anc="lm", en="b", t=2.5),
    diapo(16, en="a"))

# ---- 6. La cuenta completa
v.b("La cuenta completa, en la diapositiva diecisiete. [[a]]Datos: uno, dos, tres, ocho, nueve y diez. Grupos: uno, dos, tres, con centro dos; y ocho, nueve, diez, con centro nueve. La media global es 5.5. [[b]]WSS: uno menos dos al cuadrado, uno; dos menos dos, cero; tres menos dos, uno. Lo mismo en el otro grupo. Total: cuatro. [[c]]BSS: el centro dos está a tres coma cinco de la media global; al cuadrado, 12.25, y como el grupo tiene tres datos, 36.75. El centro nueve está a la misma distancia: otra vez 36.75. Total: 73.5.",
    CL(), T("WSS y BSS: cuenta completa · diapositiva 17", 960, 130, tam=50, c="tinta", ent="subir"),
    casillas([1, 2, 3], 270, x0=560, paso=140, c="azul", en="a", pre="ga"), casillas([8, 9, 10], 270, x0=1080, paso=140, c="ambar", en="a", pre="gb"),
    T("centro 2", 700, 360, tam=32, c="azul", w="b", en="a", t=0.4), T("centro 9", 1220, 360, tam=32, c="ambar", w="b", en="a", t=0.4),
    T("media global 5.5", 960, 420, tam=32, c="gris", w="sb", en="a", t=0.8),
    T("WSS = 1 + 0 + 1 + 1 + 0 + 1 =", 1080, 530, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(4, 1110, 530, "{:.0f}", 64, "azul", anc="lm", en="b", t=3.0),
    T("BSS = 3(2 − 5.5)² + 3(9 − 5.5)² = 36.75 + 36.75 =", 1300, 660, tam=42, c="tinta", w="sb", anc="rm", en="c"),
    N(73.5, 1330, 660, "{:.1f}", 64, "teal", anc="lm", en="c", t=4.0), diapo(17, en="a"), sec="6. La cuenta completa")
v.b("Y el TSS: [[a]]las distancias de cada dato a cinco coma cinco, al cuadrado: 20.25, 12.25, 6.25, 6.25, 12.25 y 20.25. Suman 77.5. [[b]]Verificación: cuatro más setenta y tres coma cinco es setenta y siete coma cinco. La descomposición cuadra.",
    T("TSS = 20.25 + 12.25 + 6.25 + 6.25 + 12.25 + 20.25 =", 1300, 790, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(77.5, 1330, 790, "{:.1f}", 64, "rojo", anc="lm", en="a", t=4.0),
    P("4 + 73.5 = 77.5", 960, 920, "violeta", 50, en="b"))

# ---- 7. Alerta
v.b("Aquí hay una precisión sobre la grabación. [[a]]Al explicar este WSS, se dijo: uno menos dos al cuadrado, después sacamos la raíz, va a ser uno. [[b]]El WSS no saca ninguna raíz: suma distancias al cuadrado. En este ejemplo da lo mismo, porque las diferencias son cero o uno, y la raíz de uno es uno. [[c]]Pero con una diferencia de dos, el cuadrado es cuatro y la raíz es dos. Sacar la raíz cambiaría el criterio.",
    CL(), alerta(160),
    cita("1 menos 2 al cuadrado, después sacamos la raíz, va a ser 1", y=300, en="a"),
    T("el WSS suma cuadrados: no hay raíz", 960, 450, tam=46, c="verde", w="b", en="b"),
    T("con diferencias 0 o 1 coincide: √1 = 1", 960, 550, tam=40, c="tinta", w="sb", en="b", t=2.5),
    T("con una diferencia de 2: cuadrado = 4, raíz = 2", 960, 680, tam=44, c="rojo", w="sb", en="c"),
    grabacion("0:47:22", en="a"), etiqueta("Diapositiva 17: WSS = 1 + 0 + 1 + 1 + 0 + 1 = 4 (cuadrados)", 1010, en="b"), sec="7. Alerta: sin raíz")

# ---- 8. Los vinos
v.b("Con los vinos reales y tres grupos: [[a]]el WSS es 1267.37. [[b]]El BSS, 1020.63. [[c]]Y su suma, el TSS, es 2288. Ese número no es casualidad: cada una de las trece variables estandarizadas tiene varianza uno, así que su suma de cuadrados es ciento setenta y siete menos uno, ciento setenta y seis. Trece por ciento setenta y seis da exactamente 2288.",
    CL(), T("los vinos con k = 3 · diapositiva 27", 960, 150, tam=52, c="tinta", ent="subir"),
    T("WSS", 400, 300, tam=40, c="gris", w="sb", en="a"), N(KM["wss_k3"], 400, 390, "{:.2f}", 76, "azul", en="a"),
    T("+", 680, 390, tam=70, c="gris", en="b"), T("BSS", 960, 300, tam=40, c="gris", w="sb", en="b"), N(KM["bss"], 960, 390, "{:.2f}", 76, "teal", en="b"),
    T("=", 1240, 390, tam=70, c="gris", en="c"), T("TSS", 1520, 300, tam=40, c="gris", w="sb", en="c"), N(KM["tss"], 1520, 390, "{:.0f}", 76, "rojo", en="c"),
    T("TSS = 13 variables × (177 − 1) = 13 × 176 = 2288", 960, 600, tam=44, c="violeta", w="sb", en="c", t=3.0),
    real(1050, en="a"), adicional(1010, en="c", texto="Por qué TSS = 2288: cálculo adicional"), sec="8. Los vinos")

# ---- 9. Error típico
v.b("El error típico: [[a]]comparar el WSS de dos análisis con escalas distintas, o con variables distintas, y concluir que uno agrupa mejor. [[b]]Ya viste que solo cambiar de ddof cambia el WSS de 1267.37 a 1274.57 con los mismos grupos. [[c]]El WSS solo se compara entre soluciones con los mismos datos, la misma escala y el mismo criterio.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("comparar WSS con escalas o variables distintas", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    T("mismos grupos: 1267.37 con ddof = 1 · 1274.57 con ddof = 0", 960, 460, tam=40, c="tinta", w="sb", en="b"),
    K(330, 630, "check", 70, en="c"), T("comparar solo con los mismos datos, escala y criterio", 410, 630, tam=44, c="verde", anc="lm", en="c"),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Datos: cero, dos, cuatro y diez. Grupos: cero, dos y cuatro en uno; el diez solo en el otro. La media global es cuatro. [[b]]¿Cuánto valen el WSS, el BSS y el TSS?",
    CL(), T("datos 0, 2, 4 | 10   ·   media global 4", 960, 300, tam=56, c="tinta", w="b", en="a"),
    P("WSS = ?   BSS = ?   TSS = ?", 960, 500, "violeta", 56, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Los centros son dos y diez. [[a]]WSS: cuatro más cero más cuatro, más cero: ocho. [[b]]BSS: el centro dos está a dos de la media; al cuadrado, cuatro, por tres datos, doce. El centro diez está a seis; al cuadrado, treinta y seis, por un dato. Total: 48. [[c]]TSS: dieciséis más cuatro más cero más treinta y seis: 56. Y ocho más cuarenta y ocho, cincuenta y seis.",
    CL(), T("centros: 2 y 10", 960, 200, tam=48, c="gris", w="sb"),
    T("WSS = 4 + 0 + 4 + 0 =", 1000, 340, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(8, 1030, 340, "{:.0f}", 66, "azul", anc="lm", en="a", t=1.0),
    T("BSS = 3(2 − 4)² + 1(10 − 4)² = 12 + 36 =", 1000, 470, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(48, 1030, 470, "{:.0f}", 66, "teal", anc="lm", en="b", t=2.5),
    T("TSS = 16 + 4 + 0 + 36 =", 1000, 600, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(56, 1030, 600, "{:.0f}", 66, "rojo", anc="lm", en="c", t=1.5),
    P("8 + 48 = 56", 960, 760, "violeta", 50, en="c", t=3.0), ilustrativo(1050))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]K-means minimiza el WSS, la suma de distancias euclidianas al cuadrado de cada punto a su centro. [[b]]El mejor centro de un grupo es su media. [[c]]Y la variabilidad total se reparte en dentro y entre: TSS es igual a WSS más BSS, así que bajar el WSS es subir el BSS.",
    CL(), P("K-means minimiza el WSS (cuadrados)", 960, 300, "violeta", 48, en="a"),
    P("el mejor centro es la media del grupo", 960, 460, "teal", 48, en="b"),
    P("TSS = WSS + BSS", 960, 620, "azul", 52, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]cómo encuentra K-means esos grupos, paso a paso, y por qué hay que probar muchos puntos de partida.",
    CL(), siguiente(4, "Las iteraciones de K-means y los inicios"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
