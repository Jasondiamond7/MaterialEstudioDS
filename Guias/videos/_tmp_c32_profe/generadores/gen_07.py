import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_07_cargas.json")
v = V("Leer las cargas y el efecto de la escala")
CG, SS, ES = D["cargas"], D["scree_s"], D["escala"]
COLS = D["cols"]
ORD = ["segundos", "frec_llamadas", "n_contactos", "clv", "frec_sms", "monto", "sms_por_antiguedad", "antiguedad",
       "numero_fallas", "reclamos", "edad", "duracion_prom_llamada", "intensidad_contacto", "tipo_plan_2", "estado_2"]

v.portada("Leer los componentes", "Cargas, correlaciones, y qué pasa si te olvidas de estandarizar",
          "Sexta parada: qué hay dentro de cada componente, cómo se lee, y qué pasa si no estandarizas.")

# ---- 1. Hook
v.b("Imagina un jugo que se vende como «tropical». [[a]]El nombre suena bien, pero no te dice nada. [[b]]Para saber qué tiene, lees la lista de ingredientes, y cuánto lleva de cada uno.",
    CL(), Bc(560, 520, 460, 500, "blanco", "ambar", 6, 40, sombra=True, ent="pop"), T("«tropical»", 560, 520, tam=64, c="ambar", t=0.1),
    P("el nombre no dice nada", 560, 860, "gris", 40, en="a"),
    Tj(1340, 520, 600, 500, "ingredientes", ["mango ........ 40 %", "piña .......... 35 %", "naranja ....... 20 %", "azúcar ......... 5 %"], c="violeta", en="b", id="ing"),
    ilustrativo(990, texto="analogía"), sec="1. La lista de ingredientes")
v.b("Un componente principal es igual. [[a]]PC1 es una receta. La lista de ingredientes se llama cargas: [[b]]cuánto pesa cada variable en ese componente.",
    CL(), P("componente = receta", 960, 360, "violeta", 64, en="a"), P("cargas = cuánto pesa cada variable", 960, 560, "teal", 56, en="b"))

# ---- 2. Cargas PC1 -> PC2 -> PC3
CX, SC = 1200, 500
Y0, DY = 230, 46


def barra_xs(val):
    return (CX, CX + SC * val) if val >= 0 else (CX + SC * val, CX)


def col(val):
    return "violeta" if val >= 0 else "ambar"


def crear_barras(pc, pre, signo=1):
    acts = []
    for i, nm in enumerate(ORD):
        y = Y0 + DY * i
        val = signo * CG[pc][nm]
        a, b = barra_xs(val)
        acts += [T(nm, CX - 500, y, tam=28, c="tinta", w="sb", anc="rm", id=f"{pre}n{i}", t=0.03 * i),
                 Bx(a, y - 15, b, y + 15, col(val), None, 0, 6, id=f"{pre}b{i}", t=0.03 * i, ent="fade"),
                 N(val, (b + 14) if val >= 0 else (a - 14), y, "{:+.2f}", 28, col(val), anc="lm" if val >= 0 else "rm", id=f"{pre}v{i}", t=0.03 * i, dur=0.5)]
    return acts


def a_pc(pc, pre="c", en=None, t=0.0, signo=1):
    acts = []
    for i, nm in enumerate(ORD):
        val = signo * CG[pc][nm]
        a, b = barra_xs(val)
        acts += [U(f"{pre}b{i}", x0=a, x1=b, relleno=col(val), en=en, t=t, dur=0.9),
                 U(f"{pre}v{i}", valor=val, x=(b + 14) if val >= 0 else (a - 14), ancla="lm" if val >= 0 else "rm", color=col(val), en=en, t=t, dur=0.9)]
    return acts


v.b("Estas son las cargas de PC1, en los datos de la clase. [[a]]Las seis variables de uso pesan casi lo mismo: segundos, llamadas, contactos, valor del cliente, mensajes y monto, [[b]]entre cero coma treinta y cinco, y cero coma cuarenta y tres. El resto, casi nada.",
    CL(), T("cargas de PC1 · 34.9 % de la varianza", 960, 170, tam=40, c="violeta", w="sb", id="tit"),
    L(CX, Y0 - 30, CX, Y0 + DY * 14 + 30, "grisclaro", 3, id="eje"), crear_barras("PC1", "c"),
    R(CX + 140, Y0 + DY * 2.5, 330, DY * 6 + 20, "violeta", 20, id="ruso", en="a", ent="fade"), T("«uso del servicio»", 1720, Y0 + DY * 2.5, tam=38, c="violeta", en="b", id="nuso"),
    sec="2. Las cargas de PC1, PC2 y PC3")
v.b("Por eso PC1 se puede leer como «uso del servicio»: [[a]]un cliente con PC1 alto habla más, llama más, y manda más mensajes. [[b]]Pero ese nombre lo pone quien analiza. No viene en los datos.",
    P("PC1 alto = usa más el servicio", 1720, 520, "violeta", 32, en="a", id="pal", mw=360), P("el nombre lo pones tú", 1720, 680, "ambar", 34, en="b", id="pn"))
v.b("Ahora PC2, con ocho coma nueve por ciento. [[a]]Las mismas variables, otra receta. [[b]]Pesan la antigüedad, más cero coma setenta y uno, y los SMS por antigüedad, menos cero coma sesenta.",
    X("ruso", "nuso", "pal", "pn", dur=0.3), U("tit", texto="cargas de PC2 · 8.9 % de la varianza", color="teal", dur=0.4),
    a_pc("PC2", en="a"), R(CX + 10, Y0 + DY * 6.5, 880, DY * 2 + 20, "teal", 20, id="r2", en="b", ent="fade"))
v.b("Fíjate en los signos: [[a]]uno positivo y el otro negativo. Cuando la antigüedad sube, los SMS por mes de antigüedad bajan. [[b]]Un cliente antiguo manda pocos mensajes, por mes.",
    P("signos opuestos: una sube, la otra baja", 1720, 860, "teal", 30, en="a", id="p2", mw=360))
v.b("Y PC3, con siete coma ocho por ciento. [[a]]Aquí aparecen las fallas, más cero coma setenta y dos, y los reclamos, más cero coma cincuenta. [[b]]Es el componente de la fricción con el servicio.",
    X("r2", "p2", dur=0.3), U("tit", texto="cargas de PC3 · 7.8 % de la varianza", color="rojo", dur=0.4),
    a_pc("PC3", en="a"), R(CX + 220, Y0 + DY * 8.5, 480, DY * 2 + 20, "rojo", 20, id="r3", en="b", ent="fade"), P("«fricción»", 1760, Y0 + DY * 8.5, "rojo", 38, en="b", id="p3"))
v.b("Y aquí una alerta. [[a]]La diapositiva y el apunte dicen que PC2 es fallas y reclamos. [[b]]En el notebook, fallas y reclamos forman PC3. [[c]]La regla: comprueba siempre con components_, no con el nombre de la diapositiva.",
    CL(), P("alerta del material", 960, 220, "ambar", 48, ent="subir"),
    T("«PC2 = fallas y reclamos»", 960, 380, tam=64, c="gris", id="qpc2", en="a"), U("qpc2", tachado=1.0, en="b", dur=0.6),
    P("en el notebook: fallas 0.72 y reclamos 0.50 → PC3", 960, 560, "rojo", 46, en="b", t=0.3),
    T("pd.DataFrame(pca.components_, columns=X.columns)", 960, 740, tam=36, c="tinta", w="m", en="c"), sec="3. Una alerta del material")

# ---- 4. Signo arbitrario
v.b("Un detalle que confunde mucho: el signo global de un componente es arbitrario. [[a]]Si das vuelta todas las cargas de PC1, multiplicando por menos uno, [[b]]el componente es exactamente el mismo. Lo que importa es la relación entre los signos.",
    CL(), T("cargas de PC1", 960, 170, tam=40, c="violeta", w="sb", id="tit2"), L(CX, Y0 - 30, CX, Y0 + DY * 14 + 30, "grisclaro", 3, id="eje2"),
    crear_barras("PC1", "s"), a_pc("PC1", "s", en="a", signo=-1), U("tit2", texto="cargas de PC1 × (−1)", en="a", dur=0.4),
    P("× (−1): el mismo componente", 1720, 600, "ambar", 34, en="b", mw=360), sec="4. El signo es arbitrario")

# ---- 5. Ejemplo a mano
v.b("Hagamos las cuentas con el ejemplo del apunte. [[a]]PC1 tiene cuatro cargas: cero coma cinco, cero coma cinco, cero coma cinco, y menos cero coma cinco. [[b]]Primero, comprobamos que su largo es uno: la raíz de cuatro por cero coma veinticinco, raíz de uno, igual a uno.",
    CL(), T("ejemplo del apunte · PC1 con λ = 1.6", 960, 180, tam=38, c="gris", w="sb"),
    T("cargas = (0.5 , 0.5 , 0.5 , −0.5)", 960, 320, tam=60, c="violeta", en="a"),
    T("norma = √(4 · 0.25) = √1 = 1", 960, 460, tam=56, c="tinta", en="b"), K(1500, 460, "check", 60, en="b", t=0.8), sec="5. Un ejemplo a mano")
v.b("Un cliente estandarizado es z igual a dos, uno, cero, uno. [[a]]Su puntaje en PC1 es la suma de carga por z: [[b]]cero coma cinco por dos, más cero coma cinco por uno, más cero, menos cero coma cinco por uno. [[c]]Uno, más cero coma cinco, menos cero coma cinco: uno.",
    T("cliente z = (2 , 1 , 0 , 1)", 960, 590, tam=52, c="azul", t=0.1), T("puntaje = Σ carga · z", 960, 690, tam=46, c="gris", w="sb", en="a"),
    T("0.5·2 + 0.5·1 + 0.5·0 − 0.5·1", 960, 790, tam=52, c="tinta", en="b"), N(1.0, 960, 900, "= {:.1f}", 76, "violeta", en="c", dur=0.6))
v.b("Y la correlación de una variable con el componente es su carga, por la raíz de lambda. [[a]]Para la variable uno: cero coma cinco por raíz de uno coma seis. [[b]]Cero coma cinco por uno coma dos seis cinco: cero coma sesenta y tres.",
    CL(), caja_formula("correlación = carga · √λ", y=300, c="teal", tam=60, w=1100),
    T("0.5 · √1.6 = 0.5 · 1.265", 960, 520, tam=60, c="tinta", en="a"), N(0.63, 960, 660, "= {:.2f}", 100, "teal", en="b", dur=0.6))
v.b("Con los datos de la clase: [[a]]segundos tiene carga cero coma cuatro dos cinco en PC1, y lambda uno vale cinco coma dos cuatro. [[b]]Cero coma cuatro dos cinco por raíz de cinco coma dos cuatro: cero coma noventa y siete. Segundos y PC1 casi son lo mismo.",
    CL(), T("con los datos de la clase", 960, 220, tam=40, c="gris", w="sb"),
    T("segundos: carga 0.425 · λ₁ = 5.24", 960, 380, tam=56, c="violeta", en="a"),
    T("0.425 · √5.24 = 0.425 · 2.29", 960, 530, tam=56, c="tinta", en="b"), N(D["circ"]["segundos"][0], 960, 680, "= {:.2f}", 100, "violeta", en="b", t=0.5, dur=0.6))

# ---- 6. ¿PC1 separa la fuga?
PJ = D["proy"]
pts = [(x, y, f) for x, y, f in zip(PJ["x"], PJ["y"], PJ["fuga"]) if -4.5 <= x <= 7 and -4 <= y <= 3]
v.b("¿Y la fuga? Aquí están los clientes del entrenamiento, en PC1 y PC2. [[a]]En gris, los que siguen. [[b]]En rojo, los que se fugaron. PCA no usó ese color para construir nada.",
    CL(),
    G(260, 220, 1240, 860, -4.5, 7, -4, 3, [dict(xs=[p[0] for p in pts if p[2] == 0], ys=[p[1] for p in pts if p[2] == 0], color="#94A3B8", linea=False, marcadores=7),
                                         dict(xs=[p[0] for p in pts if p[2] == 1], ys=[p[1] for p in pts if p[2] == 1], color="rojo", linea=False, marcadores=8, alfa=0.0)],
      xticks=[-4, -2, 0, 2, 4, 6], yticks=[-4, -2, 0, 2], yfmt="{:.0f}", xlabel="PC1 (34.9 %)", id="gf", t=0.2),
    T("PC2 (8.9 %)", 260, 170, tam=32, c="tinta", w="sb", anc="lm", t=0.3), P("gris: siguen", 1560, 360, "gris", 40, en="a"), P("rojo: se fugaron", 1560, 470, "rojo", 40, en="b"),
    U("gf", **{"series.1.alfa": 1.0}, en="b", dur=0.6), T(f"muestra de {len(pts)} clientes de entrenamiento", 1560, 960, tam=28, c="gris", w="sb", en="a", t=0.5),
    sec="6. ¿Qué componente se relaciona con la fuga?")
FC = D["pc_fuga_corr"]
v.b("Los rojos están mezclados con los grises: PC1 y PC2 no separan la fuga. [[a]]La correlación de PC1 con la fuga es apenas menos cero coma cero seis. [[b]]La de PC3, el componente de las fallas y los reclamos, es cero coma treinta y tres.",
    P("PC1 y PC2 no separan la fuga", 1560, 600, "gris", 36, mw=560),
    T("correlación con la fuga", 1560, 690, tam=30, c="gris", w="sb", en="a"),
    N(FC[0], 1560, 760, "PC1: {:+.2f}", 50, "violeta", en="a", t=0.2, dur=0.6), N(FC[2], 1560, 860, "PC3: {:+.2f}", 50, "rojo", en="b", dur=0.6))
v.b("Ese es el error típico de esta parada. [[a]]Pensar que PC1 es «lo más importante para predecir». [[b]]PC1 es la dirección de mayor varianza, no de mayor relación con la fuga. Varianza no es relevancia.",
    CL(), T("«PC1 es lo más importante para predecir»", 960, 330, tam=60, c="tinta", id="qimp", en="a"), U("qimp", tachado=1.0, en="b", dur=0.6),
    P("PC1 = más varianza, no más relación con la fuga", 960, 540, "rojo", 50, en="b", t=0.3), P("varianza ≠ relevancia", 960, 700, "violeta", 56, en="b", t=0.8),
    sec="7. El error típico")

# ---- 8. Sin estandarizar
CU = ES["cuota"]
v.b("Último tema: ¿qué pasa si haces PCA sin estandarizar? [[a]]PCA maximiza la varianza. Y en unidades originales, segundos concentra el noventa y nueve coma dos por ciento de toda la varianza. [[b]]El valor del cliente, cero coma seis. El resto, casi nada.",
    CL(), T("PCA sin estandarizar · cuota de la varianza total", 960, 190, tam=40, c="gris", w="sb"),
    *[[T(nm, 560, 340 + 130 * i, tam=36, c="tinta", w="sb", anc="rm", en=k), Bx(600, 310 + 130 * i, 600 + max(4, 10 * CU[nm]), 370 + 130 * i, c, None, 0, 8, grow="x", en=k, t=0.1),
       N(CU[nm], 620 + max(4, 10 * CU[nm]), 340 + 130 * i, "{:.2f}", 40, c, suf=" %", anc="lm", en=k, t=0.5, dur=0.6)]
      for i, (nm, c, k) in enumerate(zip(list(CU)[:4], ["rojo", "ambar", "gris", "gris"], ["a", "b", "b", "b"]))],
    sec="8. Sin estandarizar")
v.b("Entonces PC1, sin estandarizar, explica el noventa y nueve coma siete uno por ciento, [[a]]y es prácticamente la variable segundos: su carga es cero coma nueve nueve ocho, y la correlación, uno. [[b]]Con un solo componente ya pasas el noventa por ciento. Parece un gran resumen, pero solo refleja la unidad de medida.",
    CL(), Bc(560, 470, 640, 340, "blanco", "rojo", 5, 30, sombra=True, ent="subir"), T("sin estandarizar", 560, 360, tam=40, c="rojo"),
    N(ES["pc1_raw"], 560, 480, "PC1 = {:.2f}", 70, "rojo", suf=" %", dur=0.8, t=0.2), T(f"carga de segundos {ES['carga_seg']:.3f}", 560, 580, tam=34, c="tinta", w="sb", en="a"),
    Bc(1360, 470, 640, 340, "blanco", "violeta", 5, 30, sombra=True, ent="subir", t=0.3), T("estandarizado", 1360, 360, tam=40, c="violeta", t=0.4),
    N(SS["var"][0], 1360, 480, "PC1 = {:.1f}", 70, "violeta", suf=" %", dur=0.8, t=0.5), T("uso del servicio: 6 variables", 1360, 580, tam=34, c="tinta", w="sb", t=0.6),
    P("solo refleja la unidad de segundos", 960, 820, "rojo", 50, en="b"))

# ---- 9. Piensa
v.b("Piensa un segundo: [[a]]la antigüedad tiene carga cero coma setenta y uno en PC2, y lambda dos vale uno coma treinta y tres. ¿Cuál es su correlación con PC2?",
    CL(), T("antigüedad en PC2: carga 0.715", 960, 330, tam=60, c="teal", en="a"), T("λ₂ = 1.33", 960, 450, tam=60, c="teal", en="a", t=0.3),
    T("correlación = ?", 960, 620, tam=80, c="tinta", en="a", t=0.7), sec="9. Piensa un segundo", pausa=0.8)
v.b("Carga por raíz de lambda: [[a]]cero coma siete uno cinco por raíz de uno coma treinta y tres, que es uno coma uno cinco. [[b]]Da cero coma ochenta y dos: la antigüedad casi define PC2.",
    CL(), T("0.715 · √1.33 = 0.715 · 1.15", 960, 360, tam=60, c="tinta", en="a"), N(D["circ"]["antiguedad"][1], 960, 520, "= {:.2f}", 110, "teal", en="b", dur=0.6),
    P("la antigüedad casi define PC2", 960, 720, "teal", 52, en="b", t=0.6))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]las cargas dicen cuánto pesa cada variable, y el nombre del componente lo pones tú, leyéndolas. [[b]]El signo global es arbitrario. [[c]]Correlación es carga por raíz de lambda. [[d]]Y sin estandarizar, PC1 solo refleja la variable de mayor unidad.",
    CL(), P("cargas → tú nombras el componente", 960, 260, "violeta", 50, en="a"), P("el signo global es arbitrario", 960, 420, "ambar", 50, en="b"),
    P("correlación = carga · √λ", 960, 580, "teal", 50, en="c"), P("sin estandarizar: PC1 = la mayor unidad", 960, 740, "rojo", 50, en="d"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]t-SNE, un mapa no lineal para explorar.",
    CL(), siguiente(8, "t-SNE: un mapa para explorar"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
