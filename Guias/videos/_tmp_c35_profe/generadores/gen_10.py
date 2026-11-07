import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, alerta, k1, flujo

OUT = out("guion_c35_10_comparacion.json")
v = V("La comparación final")
CH, CA = FG["cmp_h"], FG["cmp_a"]
NH, GR = D["nb_heart"], D["nb_heart_grid_rf"]

v.portada("La comparación final", "Árbol, logística, bagging y Random Forest, con la misma vara",
          "Décima parada: la comparación final. Árbol, regresión logística, bagging y Random Forest, medidos con la misma vara. El resultado tiene una sorpresa.")

# ---- 1. Hook
v.b("Después de toda la clase, la pregunta obvia: [[a]]¿el Random Forest le gana a todo? [[b]]Vamos a medirlo con honestidad, y la respuesta va a depender de los datos.",
    CL(), T("¿el Random Forest le gana a todo?", 960, 330, tam=66, c="tinta", en="a"), P("depende de los datos", 960, 520, "ambar", 56, en="b"),
    sec="1. ¿Gana siempre?")

# ---- 2. Analogía
v.b("Una advertencia antes de mirar números. [[a]]El test de Heart tiene solo 76 pacientes. Comparar modelos con una sola partición tan chica es como decidir quién es el mejor atleta con una sola carrera, con viento a favor para algunos. [[b]]Por eso usamos validación cruzada repetida: cinco bloques, tres veces, quince carreras.",
    CL(), T("una sola carrera no decide quién es el mejor", 960, 200, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "1 partición", "76 pacientes: mucho azar", "a"), (1360, "verde", "CV 5 × 3", "15 carreras: media ± desvío", "b")], y=480, w=660, h=300),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: una sola carrera")

# ---- 3. Heart
MH = [("Árbol", "árbol", "gris"), ("Reg. logística", "regresión logística", "ambar"), ("Bagging de logísticas", "bagging de logísticas", "ambar"),
      ("Bagging de árboles", "bagging de árboles", "azul"), ("Random Forest", "Random Forest", "verde")]
v.b("Heart, AUC con validación cruzada cinco por tres. [[a]]El árbol solo: 0.753. [[b]]El bagging de árboles: 0.893. El Random Forest: 0.906. [[c]]Y la regresión logística: 0.913. La más simple de todas queda arriba.",
    CL(), T("Heart · AUC (CV 5×3) · media ± desvío", 960, 160, tam=46, c="tinta", ent="subir"),
    *[[T(lab, 640, 290 + 120 * i, tam=38, c="tinta", w="sb", anc="rm", en=m),
       Bx(660, 290 + 120 * i - 36, 660 + (CH[k][0] - 0.7) * 3600, 290 + 120 * i + 36, c, None, 0, 10, grow="x", en=m),
       T(f"{CH[k][0]:.3f} ± {CH[k][1]:.3f}", 680 + (CH[k][0] - 0.7) * 3600, 290 + 120 * i, tam=36, c=c, w="b", anc="lm", en=m, t=0.4)]
      for i, ((k, lab, c), m) in enumerate(zip(MH, ["a", "c", "c", "b", "b"]))],
    T("eje desde 0.70", 660, 900, tam=26, c="gris", w="sb", anc="lm"), real(1050), sec="3. Heart: la sorpresa")
v.b("¿Gana la logística? [[a]]Mira los desvíos: cerca de 0.035. Entre 0.906 y 0.913 hay menos de una centésima: es un empate técnico. [[b]]Lo que sí es claro: el bagging y el Random Forest rescatan al árbol, de 0.753 a cerca de 0.9. [[c]]Y en Heart, con 303 pacientes y relaciones casi lineales, la logística ya era muy buena.",
    CL(), P(f"logística {CH['Reg. logística'][0]:.3f} vs RF {CH['Random Forest'][0]:.3f}: diferencia < desvío (≈ 0.035)", 960, 260, "ambar", 42, en="a"),
    T("empate técnico", 960, 360, tam=56, c="tinta", en="a", t=0.6),
    P(f"lo claro: rescatan al árbol ({CH['Árbol'][0]:.3f} → ≈ 0.9)", 960, 510, "verde", 46, en="b"),
    T("303 pacientes, relaciones casi lineales: la logística ya era buena", 960, 650, tam=40, c="gris", w="sb", en="c"), real(1050))

# ---- 4. Ames
v.b("Ames es otro mundo: casi tres mil casas, ochenta variables, y relaciones no lineales con umbrales. [[a]]Error de prueba del árbol solo: 45 800 dólares. [[b]]Bagging: 28 900. [[c]]Random Forest con un tercio de las variables: 27 000. Aquí el bosque gana con claridad.",
    CL(), T("Ames · RMSE de prueba (586 casas)", 960, 160, tam=46, c="tinta", ent="subir"),
    *[[Bx(x - 150, 860 - val / 90, x + 150, 860, c, None, 0, 12, grow="y", en=m), N(val / 1000, x, 860 - val / 90 - 44, "{:.1f}", 56, c, suf=" mil", en=m, t=0.3),
       T(lab, x, 910, tam=34, c=c, w="sb", en=m)]
      for x, val, lab, c, m in ((560, CA["Árbol"][0], "árbol", "gris", "a"), (960, CA["Bagging"][0], "bagging", "azul", "b"),
                                (1360, CA["Random Forest (1/3)"][0], "Random Forest (1/3)", "verde", "c"))],
    L(380, 860, 1540, 860, "tinta", 4), real(1050), sec="4. Ames: gana el bosque")
v.b("Y un detalle que asusta a muchos: [[a]]en las casas de entrenamiento, el error del bosque es de solo 9 700 dólares. Casi tres veces menos que en la prueba. [[b]]¿Sobreajuste? No es un problema mientras el error de prueba sea el menor de todas las alternativas, y lo es.",
    CL(), T(f"RF · entrenamiento: {k1(CA['Random Forest (1/3)'][1])}   ·   prueba: {k1(CA['Random Forest (1/3)'][0])}", 960, 320, tam=52, c="tinta", w="b", en="a"),
    P("brecha grande ≠ problema, si la prueba es la mejor", 960, 500, "verde", 50, en="b"), real(1050))

# ---- 5. El notebook del profesor
v.b("¿Y qué dice el notebook del profesor? Lo replicamos celda por celda. [[a]]En Heart, con 120 árboles, el bagging base tiene un AUC de prueba de 0.905, y el Random Forest base, 0.928. [[b]]Tras la búsqueda, el Random Forest llega a 0.941 en la prueba. [[c]]Pero ojo: son 76 pacientes. Con otra semilla, el orden puede cambiar, y el propio notebook lo advierte.",
    CL(), T("notebook del profesor · Heart · AUC en la prueba (76 pacientes)", 960, 160, tam=42, c="tinta", ent="subir"),
    T("bagging base", 760, 320, tam=40, c="azul", w="sb", anc="rm", en="a"), N(NH["bag"]["auc"], 800, 320, "{:.3f}", 60, "azul", anc="lm", en="a", t=0.3),
    T("Random Forest base", 760, 430, tam=40, c="verde", w="sb", anc="rm", en="a", t=0.5), N(NH["rf"]["auc"], 800, 430, "{:.3f}", 60, "verde", anc="lm", en="a", t=0.8),
    T("Random Forest tras la grilla", 760, 540, tam=40, c="verde", w="sb", anc="rm", en="b"), N(GR["test_auc"], 800, 540, "{:.3f}", 60, "verde", anc="lm", en="b", t=0.3),
    P("«los ganadores pueden cambiar con otra semilla, partición o espacio de búsqueda»", 960, 720, "ambar", 38, en="c"),
    real(1050, texto="celdas del notebook replicadas"), sec="5. El notebook del profesor")
v.b("La conclusión del notebook lo dice mejor que nadie: [[a]]el ganador debe justificarse con validación cruzada y desempeño en prueba, [[b]]no por el nombre del algoritmo.",
    CL(), T("«El ganador debe justificarse con validación cruzada y desempeño en prueba…", 960, 360, tam=46, c="tinta", mw=1500, en="a"),
    P("…no por el nombre del algoritmo»", 960, 540, "violeta", 54, en="b"), T("notebook de la clase, interpretación de la competencia", 960, 680, tam=32, c="gris", w="sb", en="b", t=0.6))

# ---- 6. El flujo de trabajo (diapositiva 17)
v.b("El flujo de trabajo que propone la diapositiva diecisiete tiene cuatro pasos. [[a]]Uno, separar la prueba, antes de todo. Dos, definir la métrica. [[b]]Tres, ajustar con validación cruzada u OOB. Cuatro, evaluar una vez y explicar.",
    CL(), T("flujo de trabajo · diapositiva 17", 960, 160, tam=48, c="tinta", ent="subir"),
    flujo([("1. separar\nprueba", "teal", "a"), ("2. definir\nmétrica", "azul", "a"), ("3. ajustar con\nCV u OOB", "violeta", "b"), ("4. evaluar\ny explicar", "verde", "b")],
          y=420, w=360, h=220, x0=330, paso=420, tam=40), sec="6. El flujo de trabajo")
v.b("Y lo que hay que evitar: [[a]]filtrar información preparando los datos antes de separar; elegir la exactitud cuando la clase positiva es muy rara; [[b]]interpretar la importancia por impureza como causalidad; y ajustar mirando una y otra vez la prueba. [[c]]Como dice la nota de la diapositiva: el modelo no corrige un diseño de evaluación defectuoso.",
    Tj(960, 780, 1500, 330, "evitar", ["filtración: preparar antes de separar", "exactitud con una clase positiva muy rara",
                                       "impureza leída como causalidad", "ajustar mirando la prueba repetidamente"], c="rojo", en="a", tam=32),
    P("«el modelo no corrige un diseño de evaluación defectuoso»", 960, 1000, "ambar", 36, en="c"))

# ---- 7. Dónde encaja
v.b("¿Dónde encaja mejor? La diapositiva dieciséis da cuatro familias de aplicaciones. [[a]]Riesgo y fraude. Salud. Operaciones, como fallas o demanda. Y marketing, como respuesta o abandono. [[b]]Lo común: datos tabulares, relaciones complejas, y prioridad alta en el desempeño predictivo.",
    CL(), T("aplicaciones frecuentes · diapositiva 16", 960, 160, tam=48, c="tinta", ent="subir"),
    tarjetas([(330, "rojo", "riesgo y fraude", "transacciones, clientes", "a"), (750, "teal", "salud", "predicciones clínicas tabulares", "a"),
              (1170, "azul", "operaciones", "fallas, demanda, tiempos", "a"), (1590, "violeta", "marketing", "respuesta, abandono, valor", "a")],
             y=440, w=380, h=280, tam_t=40, tam_d=30),
    P("encaja: datos tabulares · relaciones complejas · prioridad en predecir", 960, 720, "verde", 42, en="b"), sec="7. Dónde encaja")

# ---- 8. Error típico
v.b("El error típico: [[a]]vender el Random Forest como solución universal. La nota de la diapositiva quince lo dice explícitamente: evitar venderlo así. [[b]]En Heart empató con una logística, que además se puede explicar coeficiente por coeficiente. [[c]]Primero define para qué es el modelo; después compara números con la misma vara.",
    CL(), T("«Random Forest siempre es lo mejor»", 960, 270, tam=58, c="tinta", en="a", id="e1"),
    T("nota de la diapositiva 15: «evitar vender Random Forest como solución universal»", 960, 390, tam=36, c="gris", w="sb", en="a", t=0.6),
    T("Heart: empate con una logística, que sí se puede explicar", 960, 500, tam=42, c="ambar", w="sb", en="b"),
    U("e1", tachado=1.0, color="gris", en="c", dur=0.6), P("primero: ¿para qué es? · después: la misma vara", 960, 660, "verde", 48, en="c", t=0.4),
    sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]En Heart, la logística y el Random Forest empatan. [[b]]Tienes que explicarle el modelo a un equipo médico. ¿Cuál usas?",
    CL(), P("Heart: logística ≈ Random Forest", 960, 300, "teal", 56, en="a"), T("hay que explicarlo a médicos: ¿cuál usas?", 960, 480, tam=60, c="tinta", en="b"),
    sec="9. Piensa un segundo", pausa=0.6)
v.b("La logística. [[a]]Con el mismo desempeño, gana el modelo que se puede explicar: cada coeficiente dice cómo cambia el riesgo con cada variable. [[b]]El bosque se reserva para cuando su ventaja es clara, como en Ames, o cuando solo importa predecir.",
    CL(), P("la logística: mismo desempeño, y se explica", 960, 300, "verde", 52, ent="subir"),
    T("cada coeficiente: cómo cambia el riesgo con cada variable", 960, 430, tam=42, c="gris", w="sb", en="a"),
    P("el bosque: cuando su ventaja es clara (Ames) o solo importa predecir", 960, 600, "violeta", 44, en="b"))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]bagging y Random Forest rescatan al árbol: en Heart, de 0.753 a cerca de 0.9; en Ames, de 45 800 a 27 000 dólares de error. [[b]]Contra otros modelos, el ganador depende de los datos: en Heart, empate con la logística; en Ames, gana el bosque. [[c]]Y se decide con validación cruzada y prueba, no por el nombre.",
    CL(), P(f"rescatan al árbol: {CH['Árbol'][0]:.3f} → ≈ 0.9 · {k1(CA['Árbol'][0])} → {k1(CA['Random Forest (1/3)'][0])}", 960, 300, "verde", 44, en="a"),
    P("el ganador depende de los datos: Heart empate · Ames bosque", 960, 470, "ambar", 44, en="b"), P("se decide con CV y prueba, no por el nombre", 960, 640, "violeta", 48, en="c"),
    sec="10. Para llevar")
v.b("Última parada: [[a]]el repaso final de la clase, con las preguntas de la diapositiva dieciocho y un checklist.",
    CL(), siguiente(11, "Repaso final y checklist"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
