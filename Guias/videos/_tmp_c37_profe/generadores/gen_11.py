import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, fila, codigo, alerta, matriz)

OUT = out("guion_c37_11_validacion.json")
v = V("Validación cruzada, búsqueda y AUC")
GP = D["grid_prof"]
RA_ = FG["rejilla_amplia"]
AU = D["script_auc"]
AC = D["auc_con_clases"]
PL = FG["platt"]
SC = D["script"]
TB = {(r["C"], r["gamma"]): r for r in GP["tabla"]}

v.portada("Validación cruzada y AUC", "Elegir sin tocar la prueba, y medir el orden de los puntajes",
          "Undécima parada: cómo elegir {C|, ce,} y gamma sin tocar la prueba, qué pasó cuando R y Python eligieron distinto, y qué mide de verdad la AUC.")

# ---- 1. Hook
v.b("¿De dónde salieron {C|, ce,} igual a diez y gamma 0.1? [[a]]La diapositiva cuarenta y tres dice que se fijaron de antemano, y que no se cambiaron después de ver la prueba. [[b]]Pero en un problema nuevo, nadie te regala esos valores. [[c]]Hay que elegirlos, y sin tocar la prueba. Para eso está la validación cruzada.",
    CL(), T("¿de dónde salen C = 10 y γ = 0.1?", 960, 170, tam=56, c="tinta", ent="subir"),
    P("diapositiva 43: fijados antes de mirar la prueba", 960, 360, "teal", 48, en="a"),
    P("en un problema nuevo hay que elegirlos…", 960, 500, "gris", 48, en="b"), P("…sin tocar la prueba", 960, 630, "rojo", 54, en="c"), sec="1. ¿De dónde salen?")

# ---- 2. Analogía
v.b("Piensa en cómo te preparas para un examen. [[a]]Practicas con ensayos, ajustas tu forma de estudiar según cómo te va en ellos. [[b]]Pero el examen real lo das una sola vez. [[c]]Si te mostraran el examen antes para decidir cómo estudiar, la nota dejaría de medir lo que sabes.",
    CL(), T("ensayos y examen", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "ensayos", "ajustas cómo estudiar", "a"), (960, "verde", "examen", "una sola vez", "b"),
              (1520, "rojo", "ver el examen antes", "la nota ya no mide nada", "c")], y=500, w=480, h=330, tam_t=48, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: ensayos y examen")

# ---- 3. Tres funciones de los datos
v.b("La diapositiva once lo ordena en tres etapas. [[a]]El entrenamiento estima coeficientes, escalas y vectores de soporte. [[b]]La validación compara hiperparámetros, modelos y umbrales. [[c]]Y la prueba evalúa el procedimiento que ya fue elegido. Mirar la prueba una y otra vez para decidir la convierte en validación.",
    CL(), T("tres funciones de los datos · diapositiva 11", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(320, [("entrenamiento", 520), ("estimar coeficientes, escalas, vectores de soporte", 1220)], tams=[44, 36], colores=["teal", "tinta"], en="a", fondo="tealclaro", x0=260, x1=1700, h=100),
    fila(450, [("validación", 520), ("comparar hiperparámetros, modelos y umbrales", 1220)], tams=[44, 36], colores=["violeta", "tinta"], en="b", fondo="violetaclaro", x0=260, x1=1700, h=100),
    fila(580, [("prueba", 520), ("evaluar lo que ya se eligió", 1220)], tams=[44, 36], colores=["rojo", "tinta"], en="c", fondo="rojoclaro", x0=260, x1=1700, h=100),
    P("mirar la prueba para decidir → se vuelve validación", 960, 760, "ambar", 46, en="c", t=1.5), sec="3. Tres funciones de los datos")

# ---- 4. Validación cruzada de cinco folds
cel = []
for it in range(5):
    for f in range(5):
        val = (it == f)
        cel += [Bc(620 + 200 * f, 320 + 105 * it, 180, 86, "violetaclaro" if val else "tealclaro", "violeta" if val else "teal", 3, 14,
                   en="a" if it < 2 else "b", t=0.25 * it + 0.04 * f),
                T("validar" if val else "ajustar", 620 + 200 * f, 320 + 105 * it, tam=30, c="violeta" if val else "teal", w="sb",
                  en="a" if it < 2 else "b", t=0.25 * it + 0.04 * f + 0.05)]
v.b("Así funciona la validación cruzada de cinco folds, de la diapositiva doce. [[a]]Los 398 casos de entrenamiento se parten en cinco grupos. En la primera vuelta, se ajusta con cuatro y se valida con el quinto. [[b]]Y se rota: cada grupo valida una vez. [[c]]El resultado de cada candidato es el promedio de las cinco validaciones. Y todos los candidatos usan los mismos folds.",
    CL(), T("validación cruzada de 5 folds · diapositiva 12", 960, 150, tam=48, c="tinta", ent="subir"),
    *[T(f"fold {f + 1}", 620 + 200 * f, 255, tam=30, c="gris", w="sb") for f in range(5)],
    *[T(f"vuelta {i + 1}", 480, 320 + 105 * i, tam=30, c="gris", w="sb", anc="rm") for i in range(5)], *cel,
    P("resultado = promedio de 5 validaciones · mismos folds para todos", 960, 920, "violeta", 40, en="c"), sec="4. Validación cruzada")

# ---- 5. La búsqueda en el código
v.b("En Python, el bloque opcional de la clase lo hace con GridSearchCV. [[a]]El modelo es un Pipeline: primero StandardScaler, después la SVM radial. Así la escala se reestima dentro de cada fold, sin fugas. [[b]]La rejilla prueba tres valores de C y tres de gamma: nueve combinaciones, por cinco folds, cuarenta y cinco ajustes. [[c]]Y recibe los datos sin escalar, porque el Pipeline escala por dentro.",
    CL(), T("la búsqueda en el código · diapositiva 62", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["modelo = make_pipeline(StandardScaler(), SVC(kernel=\"rbf\"))",
            "busqueda = GridSearchCV(modelo,",
            "    {\"svc__C\": [0.1, 1, 10], \"svc__gamma\": [0.01, 0.1, 1]},",
            "    cv=5, scoring=\"accuracy\")",
            "busqueda.fit(x_entrenamiento, y_entrenamiento)"], y=400, w=1700, tam=32, c="azul", en="a"),
    P("9 combinaciones × 5 folds = 45 ajustes", 960, 700, "violeta", 46, en="b"),
    T("recibe los datos SIN escalar: el Pipeline escala en cada fold", 960, 820, tam=38, c="gris", w="sb", en="c"), sec="5. La búsqueda en el código")

# ---- 6. Resultado de la búsqueda
b, s2 = TB[(10.0, 0.1)], TB[(1.0, 0.1)]
v.b("¿Qué elige? Corrimos el bloque. [[a]]Gana {C|, ce,} igual a diez y gamma 0.1, con exactitud media 0.940, y desviación entre folds de 0.020. [[b]]Segunda, C igual a uno y gamma 0.1: 0.935, con desviación 0.044. [[c]]La diferencia entre las dos es de medio punto: unos dos casos de 398. Está muy por dentro del ruido entre folds.",
    CL(), T("resultado de la búsqueda (5 folds, entrenamiento)", 960, 150, tam=48, c="tinta", ent="subir"),
    fila(280, [("puesto", 420), ("C", 700), ("γ", 900), ("exactitud media", 1220), ("sd entre folds", 1560)], tams=[34] * 5, colores=["gris"] * 5),
    fila(390, [("1.º", 420), ("10", 700), ("0.1", 900), (f"{b['media']:.3f}", 1220), (f"{b['sd']:.3f}", 1560)], tams=[44] * 5,
         colores=["verde"] * 5, en="a", fondo="verdeclaro", x0=260, x1=1720, h=96),
    fila(510, [("2.º", 420), ("1", 700), ("0.1", 900), (f"{s2['media']:.3f}", 1220), (f"{s2['sd']:.3f}", 1560)], tams=[44] * 5,
         colores=["azul"] * 5, en="b", fondo="azulclaro", x0=260, x1=1720, h=96),
    P("diferencia ≈ 0.005 → unos 2 casos de 398: dentro del ruido", 960, 690, "ambar", 44, en="c"), real(1050, en="a"), sec="6. ¿Qué elige?")

# ---- 7. Diapositiva 63: R y Python
v.b("Y aquí aparece algo que puede confundir. [[a]]Según la diapositiva sesenta y tres, la búsqueda en R eligió {C|, ce,} igual a uno y gamma 0.1, mientras que Python eligió {C|, ce,} igual a diez y gamma 0.1. [[b]]¿Contradicción? No. Los dos lenguajes no construyen los mismos folds, y usan convenciones levemente distintas para la desviación estándar. [[c]]Y en Python, esas dos combinaciones quedaron primera y segunda, a medio punto. Con otros folds, el orden puede cambiar.",
    CL(), alerta(170, texto="cómo leer la diapositiva 63"),
    fila(330, [("R (menor error medio)", 640), ("C = 1, γ = 0.1", 1260)], tams=[42, 46], colores=["teal", "tinta"], en="a", fondo="tealclaro", x0=300, x1=1620, h=100),
    fila(460, [("Python (mayor exactitud media)", 640), ("C = 10, γ = 0.1", 1260)], tams=[42, 46], colores=["azul", "tinta"], en="a", t=0.6, fondo="azulclaro", x0=300, x1=1620, h=100),
    P("folds distintos + sd con otra convención", 960, 640, "violeta", 46, en="b"),
    P("en Python: 1.º y 2.º a medio punto → no es contradicción", 960, 780, "verde", 44, en="c"), sec="7. R y Python eligen distinto")
v.b("Un dato más, para no sobreinterpretar. [[a]]Si se amplía la rejilla a seis valores de C y cinco de gamma, la validación cruzada elige otra: C igual a cien y gamma 0.01, con 0.945. [[b]]En la prueba, esa elección acierta el 94.7 por ciento, menos que el 97.1 de la configuración de la clase. [[c]]Y no pasa nada: la prueba no es para elegir. Con 171 casos, uno o tres casos de diferencia son ruido.",
    CL(), T("rejilla ampliada (6 × 5): elige C = 100, γ = 0.01", 960, 200, tam=46, c="tinta", w="sb", en="a"),
    T(f"validación cruzada: {RA_['acc_cv']:.3f}   ·   prueba: {100 * RA_['prueba_mejor']['acc']:.1f} %", 960, 330, tam=46, c="violeta", w="sb", en="b"),
    T("configuración de la clase en prueba: 97.1 %", 960, 440, tam=44, c="gris", w="sb", en="b", t=1.0),
    P("la prueba no sirve para elegir · 1 a 3 casos = ruido", 960, 620, "ambar", 46, en="c"), real(1050, en="a"))

# ---- 8. La AUC
v.b("Ahora, la AUC: el área bajo la curva ROC. [[a]]La diapositiva sesenta y cuatro la define así: es la probabilidad de que un caso positivo tenga un puntaje mayor que un caso negativo, más la mitad de la probabilidad de empate. [[b]]O sea: mide el orden de los puntajes. No mira ningún umbral.",
    CL(), T("la AUC · diapositiva 64", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("AUC = P(S₁ > S₀) + ½ · P(S₁ = S₀)", y=320, c="violeta", tam=56, w=1200, en="a", wf="sb"),
    P("mide el ORDEN de los puntajes, sin umbral", 960, 500, "teal", 50, en="b"), sec="8. La AUC")
MAL, BEN = [0.9, 0.7, 0.4], [0.5, 0.2]
pares = [(m, bb, m > bb) for m in MAL for bb in BEN]
v.b("Calculémosla a mano, con un ejemplo pequeño. [[a]]Tres malignos, con puntajes 0.9, 0.7 y 0.4; dos benignos, con 0.5 y 0.2. Hay seis pares maligno contra benigno. [[b]]En cinco de ellos, el maligno tiene el puntaje mayor. El único que falla es 0.4 contra 0.5. [[c]]La AUC es cinco sextos: 0.833.",
    T("malignos: 0.9 · 0.7 · 0.4     benignos: 0.5 · 0.2", 960, 640, tam=42, c="tinta", w="sb", en="a"),
    *[T(f"{m} > {bb}" if ok else f"{m} < {bb}", 410 + 220 * i, 760, tam=36, c="verde" if ok else "rojo", w="sb", en="b", t=0.2 * i) for i, (m, bb, ok) in enumerate(pares)],
    T("AUC = 5 / 6 =", 940, 880, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(0.833, 970, 880, "{:.3f}", 60, "violeta", anc="lm", en="c", t=0.4),
    ilustrativo(1000))

# ---- 9. AUC con datos reales
v.b("Con los datos reales, en la prueba. [[a]]La logística tiene AUC 0.9746. La SVM lineal, también 0.9746: exactamente la misma. [[b]]Pero sus exactitudes son distintas: 94.2 y 91.8 por ciento. Ordenan los casos igual de bien, y se diferencian en dónde ponen el corte. [[c]]La SVM radial llega a 0.9834.",
    CL(), T("AUC en la prueba (puntajes continuos)", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(280, [("modelo", 560), ("AUC", 1000), ("exactitud", 1400)], tams=[36] * 3, colores=["gris"] * 3),
    fila(390, [("logística", 560), (f"{AU['Logistica']:.4f}", 1000), (f"{100 * SC['Logistica']['acc']:.1f} %", 1400)], tams=[44, 50, 46], colores=["azul", "tinta", "tinta"], en="a", fondo="azulclaro", x0=340, x1=1620, h=96),
    fila(510, [("SVM lineal", 560), (f"{AU['SVM lineal']:.4f}", 1000), (f"{100 * SC['SVM lineal']['acc']:.1f} %", 1400)], tams=[44, 50, 46], colores=["ambar", "tinta", "tinta"], en="a", t=0.8, fondo="ambarclaro", x0=340, x1=1620, h=96),
    fila(630, [("SVM radial", 560), (f"{AU['SVM radial']:.4f}", 1000), (f"{100 * SC['SVM radial']['acc']:.1f} %", 1400)], tams=[44, 50, 46], colores=["verde", "tinta", "tinta"], en="c", fondo="verdeclaro", x0=340, x1=1620, h=96),
    R(1000, 450, 260, 230, "violeta", r=20, en="b"), P("misma AUC, distinta exactitud: distinto corte", 960, 800, "violeta", 44, en="b", t=0.6),
    real(1050, en="a"), sec="9. AUC con datos reales")
v.b("Y una advertencia que el propio script escribe en un comentario: [[a]]una AUC de 0.98 no significa noventa y ocho por ciento de clasificaciones correctas. [[b]]Tampoco define el umbral, ni dice si las probabilidades están bien calibradas.",
    CL(), P("AUC = 0.98 NO significa 98 % de aciertos", 960, 380, "rojo", 52, en="a"),
    T("no define el umbral · no mide calibración (diapositiva 64)", 960, 540, tam=42, c="gris", w="sb", en="b"))

# ---- 10. Alerta: ¿los scripts no calculan AUC?
v.b("Una inconsistencia del material. [[a]]La diapositiva sesenta y cuatro termina diciendo: los scripts de esta clase no calculan AUC. [[b]]Pero el bloque nueve de los dos scripts sí la calcula: en Python, con una función de scikit-learn; en R, con el paquete {pROC|pe, roc}. [[c]]Y en la clase se ejecutó. Los valores que acabas de ver son justamente los de ese bloque.",
    CL(), alerta(170), T("«Los scripts de esta clase no calculan AUC»", 960, 320, tam=48, c="tinta", w="sb", en="a", id="cita"),
    U("cita", tachado=1.0, en="b", dur=0.8),
    codigo(["# %% 9. AUC de los tres modelos", "auc_logistica = roc_auc_score(y_prueba, probabilidad)",
            "auc_lineal <- pROC::auc(y_prueba, puntaje_lineal, ...)"], y=560, w=1600, tam=32, c="teal", en="b", t=0.4),
    P("el bloque 9 de clasificadores.py y clasificadores.R sí la calcula", 960, 800, "verde", 42, en="c"),
    T("diapositiva 64 · scripts de la clase", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="c"), sec="10. Alerta: la AUC en los scripts")

# ---- 11. Puntaje no es probabilidad
v.b("Para la AUC de la SVM, el script usa decision function: el puntaje. [[a]]Ese puntaje no es una probabilidad. [[b]]Si se pide probability igual a True, scikit-learn calibra el puntaje con el método de Platt, usando una validación cruzada interna. [[c]]Y esas probabilidades pueden discrepar de predict: en la radial de la clase, en tres casos de prueba. Con pe mayor o igual que 0.5, la exactitud baja a 95.3 por ciento, frente al 97.1 de predict.",
    CL(), T("puntaje ≠ probabilidad", 960, 150, tam=52, c="tinta", ent="subir"),
    P("decision_function → puntaje (no probabilidad)", 960, 300, "rojo", 46, en="a"),
    P("SVC(probability=True) → calibración de Platt (CV interna)", 960, 440, "violeta", 44, en="b"),
    T(f"discrepancias con predict: {PL['discrepancias']} casos", 960, 590, tam=44, c="tinta", w="sb", en="c"),
    T(f"exactitud: predict {100 * PL['acc_predict']:.1f} %  ·  p ≥ 0.5 {100 * PL['acc_p05']:.1f} %", 960, 690, tam=44, c="ambar", w="sb", en="c", t=0.8),
    real(1050, en="c"), sec="11. Puntaje no es probabilidad")

# ---- 12. El error típico
v.b("El error típico, de la lista de la diapositiva cincuenta y ocho: [[a]]calcular la AUC pasándole las clases cero y uno, en vez de los puntajes. [[b]]Con las clases, la AUC de la SVM radial sale 0.964, y la de la logística, 0.934. [[c]]Con los puntajes, son 0.983 y 0.975. Con clases, la curva se reduce a un solo punto, y la AUC se subestima.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("roc_auc_score(y, clases 0/1)", 410, 230, tam=48, c="rojo", w="m", anc="lm", en="a"),
    fila(400, [("", 600), ("con clases 0/1", 1020), ("con puntajes", 1420)], tams=[36] * 3, colores=["gris"] * 3),
    fila(500, [("SVM radial", 600), (f"{AC['SVM radial']:.3f}", 1020), (f"{AU['SVM radial']:.3f}", 1420)], tams=[44, 50, 50], colores=["tinta", "rojo", "verde"], en="b", fondo="linea", x0=380, x1=1620, h=96),
    fila(620, [("logística", 600), (f"{AC['Logistica']:.3f}", 1020), (f"{AU['Logistica']:.3f}", 1420)], tams=[44, 50, 50], colores=["tinta", "rojo", "verde"], en="b", t=0.5, fondo="linea", x0=380, x1=1620, h=96),
    P("con clases, la curva es un solo punto: la AUC se subestima", 960, 800, "ambar", 44, en="c"), real(1050, en="b"), sec="12. El error típico")

# ---- 13. Piensa un segundo
v.b("Piensa un segundo. [[a]]Dos malignos con puntajes 0.8 y 0.6; dos benignos con 0.7 y 0.1. [[b]]¿Cuánto vale la AUC?",
    CL(), P("malignos: 0.8 · 0.6     benignos: 0.7 · 0.1", 960, 360, "violeta", 52, en="a"), P("AUC = ?", 960, 520, "teal", 60, en="b"),
    ilustrativo(1000), sec="13. Piensa un segundo", pausa=0.6)
v.b("Hay cuatro pares. [[a]]0.8 contra 0.7, bien; 0.8 contra 0.1, bien; 0.6 contra 0.7, mal; 0.6 contra 0.1, bien. [[b]]Tres de cuatro: [[c]]la AUC es 0.75.",
    CL(), *[T(t_, 480 + 320 * i, 380, tam=40, c=c, w="sb", en="a", t=0.4 * i) for i, (t_, c) in enumerate([("0.8 > 0.7", "verde"), ("0.8 > 0.1", "verde"), ("0.6 < 0.7", "rojo"), ("0.6 > 0.1", "verde")])],
    T("AUC = 3 / 4 =", 940, 540, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(0.75, 970, 540, "{:.2f}", 64, "violeta", anc="lm", en="c"),
    ilustrativo(1000))

# ---- 14. Para llevar
v.b("Para llevarte: [[a]]C, gamma y el umbral se eligen con validación cruzada en el entrenamiento, con la escala dentro de un Pipeline. [[b]]Diferencias de medio punto entre candidatos son ruido: por eso R y Python pueden elegir distinto. [[c]]Y la AUC mide el orden de los puntajes: se calcula con puntajes, no con clases, y no es un porcentaje de aciertos.",
    CL(), P("elegir con validación cruzada, escala en un Pipeline", 960, 300, "teal", 46, en="a"),
    P("medio punto entre candidatos = ruido", 960, 460, "ambar", 48, en="b"),
    P("AUC: orden de los puntajes · con puntajes, no clases", 960, 620, "violeta", 46, en="c"), sec="14. Para llevar")
v.b("Siguiente y última parada: [[a]]el repaso de toda la clase, y un checklist para tu próximo clasificador.",
    CL(), siguiente(12, "Repaso final y checklist"), sec="15. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
