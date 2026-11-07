import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, k1

OUT = out("guion_c35_05_varianza.json")
v = V("Por qué promediar reduce la varianza")
VB, VR, VR10 = FG["var"]["Bagging (todas)"], FG["var"]["RF (1/3)"], FG["var"]["RF (10 %)"]
var = lambda r, s2, b: r * s2 + (1 - r) * s2 / b  # noqa: E731

v.portada("Por qué promediar funciona", "La fórmula de la varianza del promedio, y su piso",
          "Quinta parada: la matemática. Una sola fórmula explica por qué el bagging funciona, y por qué tiene un límite que ningún número de árboles puede cruzar.")

# ---- 1. Hook
v.b("Si promediar árboles es tan bueno, [[a]]¿por qué no usar un millón de árboles y llegar a error cero? [[b]]Spoiler: no se puede. Hay un piso. Y entender ese piso es entender por qué existe el Random Forest.",
    CL(), T("¿un millón de árboles = error cero?", 960, 320, tam=64, c="tinta", en="a"), P("no: hay un piso", 960, 520, "rojo", 60, en="b"),
    T("y ese piso explica por qué existe el Random Forest", 960, 660, tam=42, c="gris", w="sb", en="b", t=0.6), sec="1. ¿Error cero?")

# ---- 2. Analogía
v.b("Piensa en una encuesta. [[a]]Si preguntas a mil personas que piensan de forma independiente, sus caprichos individuales se cancelan, y el promedio es muy estable. [[b]]Pero si las mil leen el mismo diario cada mañana, [[c]]comparten el mismo sesgo, y ese sesgo común no se cancela, por mucho que agregues gente.",
    CL(), T("una encuesta de 1.000 personas", 960, 170, tam=50, c="tinta", ent="subir"),
    Bc(560, 500, 640, 420, "blanco", "verde", 5, 30, sombra=True, en="a", ent="subir"), T("independientes", 560, 330, tam=44, c="verde", en="a", t=0.1),
    Pt([(48, [300, 380, 820, 560], "verde")], 11, en="a", t=0.2), T("los caprichos se cancelan", 560, 640, tam=34, c="verde", w="sb", en="a", t=0.4),
    Bc(1360, 500, 640, 420, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("leen el mismo diario", 1360, 330, tam=44, c="rojo", en="b", t=0.1),
    Pt([(48, [1100, 380, 1620, 560], "rojo")], 11, en="b", t=0.2), T("el sesgo común NO se cancela", 1360, 640, tam=34, c="rojo", w="sb", en="c"),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: la encuesta")
v.b("Con los árboles pasa lo mismo. [[a]]Lo que importa no es solo cuántos árboles promedias, [[b]]sino cuánto se parecen entre sí.",
    P("no solo CUÁNTOS árboles…", 960, 800, "azul", 48, en="a"), P("…sino cuánto se PARECEN", 960, 910, "violeta", 48, en="b"))

# ---- 3. La fórmula
v.b("La fórmula. [[a]]Cada árbol tiene una varianza, sigma al cuadrado. Dos árboles cualesquiera tienen una correlación ro entre sus predicciones. [[b]]Entonces la varianza del promedio de B árboles es: ro sigma al cuadrado, más uno menos ro, por sigma al cuadrado, sobre B.",
    CL(), T("σ² = varianza de un árbol   ·   ρ = correlación entre dos árboles   ·   B = número de árboles", 960, 220, tam=36, c="gris", w="sb", en="a"),
    caja_formula("Var(promedio) = ρσ² + (1 − ρ)σ² / B", y=420, c="azul", tam=66, en="b", w=1300), sec="3. La fórmula")
v.b("Mírala por partes. [[a]]El segundo término tiene a B abajo: con más árboles, se achica, hasta desaparecer. [[b]]Pero el primero, ro sigma al cuadrado, no depende de B. [[c]]Ese es el piso: por muchos árboles que agregues, la varianza no baja de ahí.",
    T("ρσ²", 700, 620, tam=70, c="rojo", en="b"), T("(1 − ρ)σ² / B", 1230, 620, tam=70, c="verde", en="a"),
    T("con B grande → 0", 1230, 720, tam=38, c="verde", w="sb", en="a", t=0.5), T("no depende de B: el PISO", 700, 720, tam=38, c="rojo", w="sb", en="c"),
    R(700, 620, 260, 110, "rojo", 20, 5, en="c", t=0.3))
v.b("Dos casos extremos para fijar la idea. [[a]]Si los árboles son idénticos, ro vale uno, y la varianza es sigma al cuadrado: promediar no sirve de nada. [[b]]Si fueran independientes, ro cero, la varianza sería sigma al cuadrado sobre B: cada árbol extra ayuda.",
    CL(), T("los dos extremos", 960, 170, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "ρ = 1", "árboles idénticos: Var = σ² · promediar no sirve", "a"),
              (1360, "verde", "ρ = 0", "independientes: Var = σ²/B · cada árbol ayuda", "b")], y=480, w=660, h=340, tam_t=72))
v.b("La nota de la diapositiva nueve aclara que la fórmula supone varianzas parecidas y una correlación común. [[a]]Y una precisión útil: ro es la correlación entre las predicciones de dos árboles para un mismo caso, que es lo mismo que la correlación de sus errores en ese caso. [[b]]La diapositiva la llama correlación entre errores: es lo mismo.",
    CL(), T("nota de la diapositiva 9: supone varianzas similares y correlación común", 960, 260, tam=38, c="gris", w="sb", ent="subir"),
    P("ρ = correlación entre las predicciones de dos árboles (para un mismo caso)", 960, 420, "azul", 42, en="a"),
    T("= correlación de sus errores en ese caso (diapositiva: «correlación entre errores»)", 960, 540, tam=36, c="tinta", w="sb", en="b"))

# ---- 4. Ejemplo resuelto
v.b("Ahora un ejemplo con números, del apunte. [[a]]Sigma al cuadrado igual a 64. Primero con ro igual a 0.5. [[b]]Con un árbol: 32 más 32, igual a 64. [[c]]Con cuatro árboles: 32 más 32 sobre cuatro, que es 8: total 40. [[d]]Con dieciséis: 32 más 2, igual a 34. Y con infinitos árboles, 32.",
    CL(), T("ejemplo del apunte · σ² = 64", 960, 150, tam=46, c="tinta", ent="subir"),
    T("B", 360, 260, tam=40, c="gris", en="a"), T("ρ = 0.5", 900, 260, tam=44, c="rojo", en="a"), T("ρ = 0.1", 1500, 260, tam=44, c="verde", en="a"),
    *[[T(b, 360, 370 + 110 * i, tam=44, c="tinta", en=m), T(f"{e1}", 900, 370 + 110 * i, tam=44, c="tinta", w="sb", en=m, t=0.2)]
      for i, (b, e1, m) in enumerate([("1", "32 + 32 = 64", "b"), ("4", "32 + 8 = 40", "c"), ("16", "32 + 2 = 34", "d"), ("∞", "32", "d")])],
    ilustrativo(1000, texto="ejemplo del apunte"), sec="4. Un ejemplo resuelto")
v.b("Ahora con ro igual a 0.1, árboles menos parecidos. [[a]]Con un árbol, igual: 6.4 más 57.6, 64. [[b]]Con cuatro: 6.4 más 14.4, igual a 20.8. [[c]]Con dieciséis, 6.4 más 3.6, igual a 10. Y el piso es 6.4: cinco veces más bajo.",
    *[[T(e2, 1500, 370 + 110 * i, tam=44, c="verde", w="sb", en=m, t=0.2)]
      for i, (e2, m) in enumerate([("6.4 + 57.6 = 64", "a"), ("6.4 + 14.4 = 20.8", "b"), ("6.4 + 3.6 = 10", "c"), ("6.4", "c")])],
    R(900, 700, 260, 90, "rojo", 16, 5, en="c", t=0.6), R(1500, 700, 220, 90, "verde", 16, 5, en="c", t=0.8))
v.b("La lección del ejemplo: [[a]]con ro igual a 0.5, pasar de dieciséis árboles a infinitos solo baja la varianza de 34 a 32. [[b]]Con ro igual a 0.1, el piso entero es cinco veces menor. [[c]]Árboles que se parecen ya no aportan: hay que bajar ro, no solo subir B.",
    CL(), P("ρ = 0.5:  de 16 a ∞ árboles, solo 34 → 32", 960, 300, "rojo", 50, en="a"),
    P("ρ = 0.1:  piso 6.4, cinco veces menor", 960, 450, "verde", 50, en="b"), T("hay que bajar ρ, no solo subir B", 960, 620, tam=56, c="tinta", en="c"))

# ---- 5. Medido en Ames
v.b("¿Se cumple con datos reales? Lo medimos en Ames. [[a]]Entrenamos el mismo bosque sesenta veces, cada vez con otra submuestra de casas, y medimos cuánto varían las predicciones en un conjunto de validación. [[b]]Un árbol solo tiene una desviación de unos 36 600 dólares. Y la correlación entre dos árboles del bagging resultó ro igual a 0.049.",
    CL(), T("Ames · varianza medida (60 repeticiones)", 960, 160, tam=46, c="tinta", ent="subir"),
    T("60 entrenamientos con submuestras distintas → ¿cuánto varía la predicción?", 960, 270, tam=36, c="gris", w="sb", en="a"),
    T("σ (un árbol) ≈", 900, 420, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(VB["sigma"] / 1000, 930, 420, "{:.1f}", 66, "azul", suf=" mil", anc="lm", en="b", t=0.3),
    T("ρ (bagging) =", 900, 540, tam=50, c="tinta", w="sb", anc="rm", en="b", t=0.8), N(VB["rho"], 930, 540, "{:.3f}", 66, "azul", anc="lm", en="b", t=1.1),
    real(1050), sec="5. Medido en Ames")
v.b("Con esos dos números, la fórmula predice la desviación del promedio. [[a]]Con diez árboles: 13 900 dólares. Y lo medido fue 13 900. [[b]]Con cien árboles, la fórmula dice 8 800, y lo medido fue 8 800. [[c]]Y el piso, la raíz de ro sigma al cuadrado, es 8 100 dólares: ahí se detiene el bagging.",
    T("B", 560, 680, tam=36, c="gris", en="a"), T("fórmula", 960, 680, tam=36, c="azul", en="a"), T("medido", 1360, 680, tam=36, c="verde", en="a"),
    *[[T(str(b), 560, 770 + 90 * i, tam=42, c="tinta", en=m), T(k1(VB["formula"][str(b)]), 960, 770 + 90 * i, tam=42, c="azul", w="b", en=m, t=0.2),
       T(k1(VB["medido"][str(b)]), 1360, 770 + 90 * i, tam=42, c="verde", w="b", en=m, t=0.4)] for i, (b, m) in enumerate([(10, "a"), (100, "b")])],
    P(f"piso √(ρσ²) ≈ {k1(VB['piso'])}", 1640, 560, "rojo", 40, en="c"))

# ---- 6. Bajar ρ
v.b("¿Y si bajamos ro? [[a]]Un Random Forest que mira un tercio de las variables en cada corte tiene ro igual a 0.024, y su piso baja a 5 800 dólares. [[b]]Mirando solo el 10 % de las variables, ro baja a 0.011, y el piso a 4 400. [[c]]Cada árbol es un poco peor, sigma sube, pero se parecen mucho menos, y el promedio gana.",
    CL(), T("bajar ρ baja el piso · Ames", 960, 160, tam=48, c="tinta", ent="subir"),
    *[[T(lab, 560, 320 + 170 * i, tam=40, c=c, w="b", anc="rm", en=m), T(f"ρ = {d['rho']:.3f}", 720, 320 + 170 * i, tam=40, c="tinta", w="sb", anc="lm", en=m, t=0.2),
       T(f"σ ≈ {k1(d['sigma'])}", 1030, 320 + 170 * i, tam=40, c="gris", w="sb", anc="lm", en=m, t=0.3),
       Bx(1350, 320 + 170 * i - 35, 1350 + d["piso"] / 25, 320 + 170 * i + 35, c, None, 0, 10, grow="x", en=m, t=0.4),
       T(k1(d["piso"]), 1370 + d["piso"] / 25, 320 + 170 * i, tam=38, c=c, w="b", anc="lm", en=m, t=0.7)]
      for i, (lab, d, c, m) in enumerate([("bagging (todas)", VB, "azul", None), ("RF (1/3)", VR, "teal", "a"), ("RF (10 %)", VR10, "violeta", "b")])],
    T("piso √(ρσ²)", 1500, 230, tam=32, c="rojo", w="sb"),
    P("cada árbol un poco peor (σ sube), pero mucho menos parecidos (ρ baja)", 960, 830, "verde", 40, en="c"), real(1050), sec="6. Bajar ρ")
v.b("Eso es exactamente lo que dice la diapositiva nueve: [[a]]si ro es alto, muchos árboles repiten errores parecidos, y aumentar B no elimina ese componente. [[b]]Esa limitación es la que motiva el Random Forest.",
    CL(), T("diapositiva 9 · límite importante", 960, 260, tam=42, c="gris", w="sb", ent="subir"),
    T("«Si ρ es alto, muchos árboles repiten errores parecidos. Aumentar B no elimina ese componente.»", 960, 420, tam=46, c="tinta", mw=1500, en="a"),
    P("→ motiva el Random Forest", 960, 620, "violeta", 52, en="b"))

# ---- 7. Error típico
v.b("El error típico: [[a]]subir y subir el número de árboles esperando que el error siga bajando. [[b]]En Ames, con cien árboles el bagging ya está en 8 800, a un paso de su piso de 8 100. [[c]]Más árboles no rompen el piso: lo que lo baja es que los árboles se parezcan menos.",
    CL(), T("«si agrego más árboles, el error sigue bajando»", 960, 280, tam=54, c="tinta", en="a", id="e1"),
    T(f"B = 100: {k1(VB['medido']['100'])}  ·  piso: {k1(VB['piso'])}", 960, 420, tam=48, c="rojo", w="b", en="b"),
    U("e1", tachado=1.0, color="gris", en="c", dur=0.6), P("lo que baja el piso: árboles menos parecidos", 960, 600, "verde", 48, en="c", t=0.4),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo, con sigma al cuadrado igual a 64. [[a]]Opción uno: ro igual a 0.5, y pasas de dieciséis árboles a mil. [[b]]Opción dos: te quedas con dieciséis árboles, pero bajas ro a 0.1. ¿Cuál reduce más la varianza?",
    CL(), P("σ² = 64", 960, 200, "gris", 44, ent="subir"),
    tarjetas([(560, "rojo", "opción 1", "ρ = 0.5 · de 16 a 1.000 árboles", "a"), (1360, "verde", "opción 2", "16 árboles · ρ baja a 0.1", "b")], y=470, w=660, h=300),
    T("¿cuál reduce más?", 960, 760, tam=64, c="tinta", en="b", t=0.5), sec="8. Piensa un segundo", pausa=0.6)
v.b("Opción uno: 32 más 32 sobre mil, igual a 32.03. [[a]]Opción dos: 6.4 más 57.6 sobre dieciséis, igual a 10. [[b]]Gana por lejos bajar ro. Mil árboles parecidos valen menos que dieciséis distintos.",
    CL(), T(f"opción 1:  32 + 32/1000 = {var(0.5, 64, 1000):.2f}", 960, 300, tam=54, c="rojo", w="b", ent="subir"),
    T(f"opción 2:  6.4 + 57.6/16 = {var(0.1, 64, 16):.0f}", 960, 430, tam=54, c="verde", w="b", en="a"),
    P("16 árboles distintos > 1.000 parecidos", 960, 620, "violeta", 54, en="b"), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]la varianza del promedio es ro sigma al cuadrado, más uno menos ro, sigma al cuadrado sobre B. [[b]]Más árboles borran el segundo término, pero el piso, ro sigma al cuadrado, queda. [[c]]Para bajar el piso hay que bajar ro: árboles menos parecidos. Esa es la idea del Random Forest.",
    CL(), caja_formula("Var = ρσ² + (1 − ρ)σ² / B", y=280, c="azul", tam=56, w=1100, en="a"),
    P("más árboles: borran el 2.º término · el piso ρσ² queda", 960, 470, "rojo", 46, en="b"), P("bajar el piso = bajar ρ → Random Forest", 960, 630, "violeta", 50, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el Random Forest, y cómo baja ro con azar en las variables.",
    CL(), siguiente(6, "Random Forest: azar en las variables"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
