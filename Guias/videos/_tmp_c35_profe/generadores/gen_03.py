import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, flujo, caja_formula, codigo, mils, k1

OUT = out("guion_c35_03_oob.json")
v = V("Fuera de bolsa: el error OOB")
OC = FG["oobcv"]
H, A = OC["heart"], OC["ames"]

v.portada("Fuera de bolsa: el error OOB", "Una validación que viene de regalo con el bootstrap",
          "Tercera parada: el error fuera de bolsa, el OOB. Vas a calcularlo a mano, y vas a ver por qué es una validación casi gratis.")

# ---- 1. Hook
v.b("Para saber si un modelo generaliza, normalmente apartas datos: un conjunto de validación, o haces validación cruzada. [[a]]Eso cuesta: o pierdes datos para entrenar, o entrenas cinco veces. [[b]]¿Y si cada árbol trajera su propia validación incluida?",
    CL(), T("evaluar un modelo, la forma habitual", 960, 190, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "gris", "apartar validación", "menos datos para entrenar", "a"), (1360, "gris", "validación cruzada", "entrenar 5 veces", "a")], y=450, w=640, h=260),
    P("¿y si cada árbol trajera su validación incluida?", 960, 760, "violeta", 50, en="b"), sec="1. Validar cuesta")

# ---- 2. Analogía
v.b("Piensa en un curso donde cada alumno estudia con una guía distinta, sacada al azar del mismo banco de preguntas. [[a]]Para evaluar a un alumno, no le preguntas lo que ya estudió: [[b]]le tomas el examen con las preguntas que NO venían en su guía.",
    CL(), T("un banco de preguntas, guías distintas", 960, 170, tam=50, c="tinta", ent="subir"),
    Bc(560, 470, 560, 330, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("lo que estudió", 560, 360, tam=44, c="azul", en="a", t=0.1),
    T("no sirve para evaluarlo", 560, 470, tam=36, c="gris", w="sb", en="a", t=0.3), K(560, 560, "x", 60, en="a", t=0.5),
    Bc(1360, 470, 560, 330, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("lo que NO estudió", 1360, 360, tam=44, c="verde", en="b", t=0.1),
    T("su examen justo", 1360, 470, tam=36, c="gris", w="sb", en="b", t=0.3), K(1360, 560, "check", 60, en="b", t=0.5),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el examen justo")
v.b("Cada árbol es un alumno, [[a]]su muestra bootstrap es su guía, [[b]]y las filas que quedaron afuera, ese 36.8 %, son su examen.",
    P("árbol = alumno", 560, 800, "violeta", 44, en="a"), P("filas fuera de bolsa = su examen", 1300, 800, "teal", 44, en="b"))

# ---- 3. Cómo se calcula
v.b("La receta, tal como la dice la diapositiva seis. [[a]]Cada árbol se entrena con su muestra bootstrap. [[b]]Las filas que quedaron afuera de esa muestra le sirven de validación interna. [[c]]Y al final, cada fila recibe predicciones solo de los árboles que no la vieron.",
    CL(), T("el error OOB, paso a paso", 960, 170, tam=50, c="tinta", ent="subir"),
    flujo([("bootstrap del árbol b", "azul", "a"), ("árbol b", "violeta", "a"), ("filas OOB del árbol b", "teal", "b"), ("predicción y error OOB", "verde", "c")],
          y=470, w=340, h=200, x0=330, paso=420, tam=38), sec="3. La receta")
v.b("¿Cuántos árboles opinan sobre cada fila? [[a]]Una fila queda afuera con probabilidad 0.368, así que con B árboles, la predice más o menos un 36.8 % de ellos. [[b]]Con los 120 árboles del notebook del profesor, unos 44 árboles por fila. [[c]]Con 500, unos 184.",
    CL(), caja_formula("árboles que opinan sobre una fila ≈ 0.368 × B", y=280, c="teal", tam=54, w=1300, en="a"),
    T("B = 120 (notebook):  0.368 × 120 =", 1000, 480, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(0.368 * 120, 1030, 480, "{:.0f}", 64, "teal", anc="lm", en="b", t=0.5),
    T("B = 500:  0.368 × 500 =", 1000, 600, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(0.368 * 500, 1030, 600, "{:.0f}", 64, "teal", anc="lm", en="c", t=0.5))

# ---- 4. Ejemplo a mano
F = ["P1", "P2", "P3", "P4"]
Y = ["Sí", "No", "Sí", "No"]
# votos OOB por fila: (arbol, voto)
VOT = {"P1": [("T2", "Sí"), ("T4", "Sí")], "P2": [("T1", "No"), ("T3", "No")], "P3": [("T2", "No"), ("T3", "No"), ("T5", "Sí")],
       "P4": [("T1", "Sí"), ("T4", "No"), ("T5", "No")]}
XC = {"T1": 700, "T2": 880, "T3": 1060, "T4": 1240, "T5": 1420}
v.b("Ahora a mano, con un ejemplo pequeño. [[a]]Cuatro pacientes y cinco árboles. En la tabla, cada casilla con voto es un árbol que NO vio a ese paciente. [[b]]Las casillas vacías son árboles que sí lo vieron, y por eso no opinan.",
    CL(), T("ejemplo a mano · 4 pacientes, 5 árboles", 960, 150, tam=46, c="tinta", ent="subir"),
    T("real", 480, 250, tam=34, c="gris", w="sb", en="a"), *[T(t, x, 250, tam=36, c="violeta", en="a") for t, x in XC.items()],
    *[[T(f, 300, 350 + 110 * i, tam=40, c="tinta", en="a"), T(Y[i], 480, 350 + 110 * i, tam=38, c="tinta", w="sb", en="a")] for i, f in enumerate(F)],
    *[Bc(x, 350 + 110 * i, 150, 84, "linea", None, 0, 14, en="b", t=0.03 * j) for i in range(4) for j, (tn, x) in enumerate(XC.items())
      if tn not in dict(VOT[F[i]])],
    *[[Bc(XC[t], 350 + 110 * i, 150, 84, "blanco", "teal", 4, 14, en="a", t=0.4 + 0.1 * k),
       T(vt, XC[t], 350 + 110 * i, tam=36, c=("verde" if vt == "Sí" else "rojo"), en="a", t=0.45 + 0.1 * k)]
      for i, f in enumerate(F) for k, (t, vt) in enumerate(VOT[f])],
    ilustrativo(1000), sec="4. El OOB a mano")
v.b("Votamos fila por fila, solo con los árboles que no la vieron. [[a]]{P1|Pe uno}: dos votos Sí. Predice Sí, y acierta. [[b]]{P2|Pe dos}: dos votos No. Acierta. [[c]]{P3|Pe tres}: dos No contra un Sí. Predice No, pero el paciente era Sí: error. [[d]]{P4|Pe cuatro}: dos No contra un Sí. Predice No, y acierta.",
    T("voto OOB", 1640, 250, tam=34, c="teal", en="a"),
    *[[T(pr, 1620, 350 + 110 * i, tam=38, c="tinta", w="b", en=m), K(1730, 350 + 110 * i, "check" if ok else "x", 48, en=m, t=0.4)]
      for i, (pr, ok, m) in enumerate([("Sí", True, "a"), ("No", True, "b"), ("No", False, "c"), ("No", True, "d")])],
    R(1680, 350 + 110 * 2, 260, 96, "rojo", 16, 5, en="c", t=0.3))
v.b("Una fila mal clasificada de cuatro. [[a]]El error OOB es uno sobre cuatro, 25 %, y la exactitud OOB, 0.75. [[b]]Y fíjate: ningún árbol evaluó a un paciente que hubiera visto en su entrenamiento.",
    P("error OOB = 1/4 = 25 % · exactitud OOB = 0.75", 960, 830, "teal", 46, en="a"),
    T("ningún árbol evaluó a un paciente que vio", 960, 930, tam=38, c="gris", w="sb", en="b"))

# ---- 5. Con pocos árboles
v.b("Un detalle práctico: con muy pocos árboles, algunas filas no tienen ninguna predicción OOB. [[a]]Una fila entra a una muestra con probabilidad 0.632. Que entre a las diez muestras de diez árboles: 0.632 elevado a diez, más o menos 1 %. [[b]]Con treinta árboles, ya es prácticamente cero.",
    CL(), T("¿una fila sin ninguna predicción OOB?", 960, 190, tam=50, c="tinta", ent="subir"),
    T("B = 10:   0.632¹⁰ =", 960, 380, tam=50, c="tinta", w="sb", anc="rm", en="a"), N(0.632 ** 10 * 100, 990, 380, "{:.1f}", 64, "ambar", suf=" %", anc="lm", en="a", t=0.5),
    T("B = 30:   0.632³⁰ ≈", 960, 520, tam=50, c="tinta", w="sb", anc="rm", en="b"), T("0.0001 %", 990, 520, tam=64, c="verde", anc="lm", en="b", t=0.5),
    P("unas decenas de árboles bastan para que todas tengan OOB", 960, 720, "teal", 44, en="b", t=0.9), sec="5. Con pocos árboles")

# ---- 6. Datos reales
v.b("Ahora con datos reales: ¿se parece el OOB a las otras formas de validar? [[a]]Heart, un bosque de quinientos árboles, exactitud. El OOB da 0.811. [[b]]La validación cruzada de cinco bloques da 0.798. [[c]]Y el conjunto de prueba, 76 pacientes que nadie tocó, da 0.855.",
    CL(), T("Heart · Random Forest de 500 árboles · exactitud", 960, 160, tam=44, c="tinta", ent="subir"),
    *[[Bx(x - 140, 860 - (val - 0.7) * 3000, x + 140, 860, c, None, 0, 12, grow="y", en=m), N(val, x, 860 - (val - 0.7) * 3000 - 44, "{:.3f}", 56, c, en=m, t=0.3),
       T(lab, x, 910, tam=34, c=c, w="sb", en=m)]
      for x, val, lab, c, m in ((560, H["oob"], "OOB · 1 ajuste", "ambar", "a"), (960, H["cv"], "CV 5 bloques · 5 ajustes", "azul", "b"),
                                (1360, H["test"], "prueba · 76 pacientes", "teal", "c"))],
    L(380, 860, 1540, 860, "tinta", 4), T("eje desde 0.70", 1700, 860, tam=24, c="gris", w="sb"), real(1050), sec="6. Con datos reales")
v.b("Las tres en el mismo rango. [[a]]Y con un desvío de la validación cruzada de cerca de 0.05, esas diferencias son ruido, no un problema. [[b]]En Ames, el error en dólares cuenta la misma historia: [[c]]OOB, 26 000; validación cruzada, 26 300; prueba, 27 000 dólares de RMSE.",
    CL(), P(f"desvío de la CV ≈ {H['cv_sd']:.2f}: las diferencias son ruido", 960, 230, "ambar", 44, en="a"),
    T("Ames · Random Forest de 300 árboles · RMSE (dólares)", 960, 360, tam=42, c="tinta", en="b"),
    *[[T(lab, x, 500, tam=36, c=c, w="sb", en="c", t=0.2 * j), N(val / 1000, x, 600, "{:.1f}", 76, c, suf=" mil", en="c", t=0.2 * j + 0.2)]
      for j, (x, val, lab, c) in enumerate(((560, A["oob"], "OOB", "ambar"), (960, A["cv"], "CV 5 bloques", "azul"), (1360, A["test"], "prueba", "teal")))],
    real(1050))
v.b("Y la gran diferencia está en el costo: [[a]]el OOB salió de un solo ajuste del bosque. [[b]]La validación cruzada necesitó cinco. [[c]]Por eso el OOB es tan útil para comparar configuraciones mientras ajustas.",
    T("OOB: 1 ajuste", 660, 800, tam=50, c="ambar", en="a"), T("CV: 5 ajustes", 1260, 800, tam=50, c="azul", en="b"),
    P("ideal para comparar configuraciones mientras ajustas", 960, 920, "verde", 44, en="c"))

# ---- 7. En Python
v.b("En scikit-learn basta un parámetro. [[a]]oob_score igual a True, al crear el bosque o el bagging. [[b]]Y después de entrenar, el atributo oob_score_ trae la exactitud OOB en clasificación, o el R cuadrado OOB en regresión.",
    CL(), T("OOB en Python", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["rf = RandomForestClassifier(n_estimators=500,", "                            oob_score=True, random_state=42)", "rf.fit(X_train, y_train)",
            "rf.oob_score_          # exactitud OOB (clasificación)", "rf.oob_decision_function_   # probabilidades OOB por fila"], y=480, w=1500, tam=36, en="a", dt=0.35),
    P("regresión: oob_score_ = R² OOB", 960, 800, "teal", 44, en="b"), sec="7. En Python")

# ---- 8. Error típico
v.b("El error típico: [[a]]usar el OOB como si reemplazara al conjunto de prueba final. [[b]]La propia nota de la diapositiva seis lo advierte: el OOB ayuda a ajustar el modelo, [[c]]pero un conjunto de prueba independiente sigue siendo la referencia final, cuando está disponible.",
    CL(), T("«ya tengo el OOB, no necesito test»", 960, 270, tam=56, c="tinta", en="a", id="e1"),
    T("nota de la diapositiva 6:", 960, 420, tam=38, c="gris", w="sb", en="b"), T("«el error OOB ayuda a ajustar el modelo…»", 960, 500, tam=44, c="tinta", w="sb", en="b", t=0.3),
    U("e1", tachado=1.0, color="gris", en="c", dur=0.6), P("«…un conjunto de prueba independiente sigue siendo la referencia final»", 960, 660, "verde", 40, en="c", t=0.3),
    sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]En Ames, con trescientos árboles, el error OOB sobre las casas de entrenamiento es 27 300 dólares, [[b]]pero el error del mismo bosque en un conjunto de validación aparte es 23 100. [[c]]¿Por qué el OOB sale más alto?",
    CL(), P(f"OOB (300 árboles): {k1(FG['curvas']['oob_rf']['300'])}", 960, 280, "ambar", 54, en="a"),
    P(f"validación aparte: {k1(FG['curvas']['rf']['300'])}", 960, 420, "teal", 54, en="b"), T("¿por qué el OOB es más alto?", 960, 600, tam=66, c="tinta", en="c"),
    real(1050), sec="9. Piensa un segundo", pausa=0.6)
v.b("Porque cada fila se predice solo con los árboles que no la vieron: [[a]]más o menos 110 de los 300. [[b]]Es como evaluar un bosque más chico, y un bosque más chico es un poco peor. [[c]]Por eso el OOB tiende a ser levemente pesimista: es una estimación prudente.",
    CL(), T("cada fila: ≈ 0.368 × 300 ≈ 110 árboles", 960, 300, tam=54, c="tinta", ent="subir", en="a"),
    T("= evaluar un bosque más chico", 960, 430, tam=48, c="gris", w="sb", en="b"), P("el OOB es levemente pesimista: estimación prudente", 960, 600, "verde", 48, en="c"))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]cada fila se predice solo con los árboles que no la vieron, más o menos un 36.8 % de ellos. [[b]]Ese error OOB sale de un solo ajuste y se parece a la validación cruzada. [[c]]Úsalo para ajustar; el conjunto de prueba queda para el final.",
    CL(), P("OOB: cada fila, solo los árboles que no la vieron", 960, 300, "teal", 48, en="a"),
    P(f"1 ajuste · Heart {H['oob']:.3f} vs CV {H['cv']:.3f}", 960, 470, "ambar", 50, en="b"), P("ajustar con OOB · el test, al final", 960, 640, "verde", 50, en="c"),
    sec="10. Para llevar")
v.b("Siguiente parada: [[a]]juntar los árboles. Cómo vota y promedia el bagging.",
    CL(), siguiente(4, "Bagging: votar y promediar"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
