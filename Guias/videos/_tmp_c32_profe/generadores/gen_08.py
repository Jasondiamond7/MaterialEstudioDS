import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_08_tsne.json")
v = V("t-SNE: un mapa para explorar")
TS = D["tsne"]
TW = TS["tw"]
FU = TS["idx_fuga"]


def norm(emb):
    xs, ys = [p[0] for p in emb], [p[1] for p in emb]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    m = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    return [((x - cx) / m, (y - cy) / m) for x, y in zip(xs, ys)]


E = {k: norm(e) for k, e in TS["emb"].items()}
ORDEN = sorted(range(len(FU)), key=lambda i: E["PCA"][i][0])        # orden fijo de los puntos (por x del PCA)
I0 = [i for i in ORDEN if FU[i] == 0]
I1 = [i for i in ORDEN if FU[i] == 1]
DUMMY = 50.0                                                         # punto fuera del lienzo: hace visible toda la serie


def serie(k, idx):
    return [E[k][i][0] for i in idx] + [DUMMY], [E[k][i][1] for i in idx] + [DUMMY]


def mapa(k, id_, x0=360, y0=230, lado=620, t=0.0, en=None, alfa_fuga=1.0):
    x0_, y0_ = serie(k, I0)
    x1_, y1_ = serie(k, I1)
    return [Bc(x0 + lado / 2, y0 + lado / 2, lado + 60, lado + 60, "blanco", "linea", 3, 24, en=en, t=t, id=f"{id_}m", ent="fade"),
            G(x0, y0, x0 + lado, y0 + lado, -1.08, 1.08, -1.08, 1.08,
              [dict(xs=x0_, ys=y0_, color="#94A3B8", linea=None, marcadores=6), dict(xs=x1_, ys=y1_, color="rojo", linea=None, marcadores=7, alfa=alfa_fuga)],
              xticks=[], yticks=[], rejilla=False, ejes=0.0, id=id_, en=en, t=t + 0.1)]


def morph(id_, k, en=None, t=0.0, dur=1.6):
    x0_, y0_ = serie(k, I0)
    x1_, y1_ = serie(k, I1)
    return U(id_, **{"series.0.xs": x0_, "series.0.ys": y0_, "series.1.xs": x1_, "series.1.ys": y1_}, en=en, t=t, dur=dur)


v.portada("t-SNE", "Un mapa no lineal de quién es vecino de quién: para explorar, no para producir",
          "Séptima parada: t-SNE. Un mapa no lineal para explorar tus datos, y por qué no se usa para producir variables.")

# ---- 1. Hook
v.b("Imagina que organizas las mesas de un matrimonio. [[a]]Tu única regla: los que se conocen, se sientan juntos. [[b]]Al final, cada mesa tiene amigos. Pero que la mesa tres esté cerca de la puerta, o lejos de la mesa siete, no significa nada.",
    CL(), P("las mesas de un matrimonio", 960, 220, "violeta", 54, ent="subir"),
    *[[Bc(x, y, 180, 180, "blanco", c, 5, 90, en="a", t=0.15 * j, ent="pop"),
       Pt([(5, [x - 55, y - 40, x + 55, y + 40], c)], 11, id=f"me{j}", en="a", t=0.15 * j + 0.1)]
      for j, (x, y, c) in enumerate(((420, 500, "azul"), (760, 760, "teal"), (1180, 470, "ambar"), (1520, 760, "violeta")))],
    L(500, 560, 1100, 520, "rojo", 4, guion=True, en="b", t=0.6), T("¿distancia entre mesas? no significa nada", 960, 950, tam=40, c="rojo", w="sb", en="b", t=0.9),
    ilustrativo(1010, texto="analogía"), sec="1. Las mesas de un matrimonio")
v.b("Eso es t-SNE. [[a]]Dibuja los datos en dos dimensiones, cuidando quién es vecino de quién. [[b]]A diferencia de PCA, no le importa la varianza total. Solo las vecindades.",
    CL(), P("t-SNE: un mapa de vecindades", 960, 330, "violeta", 62, en="a"), P("PCA cuida la varianza global · t-SNE cuida los vecinos", 960, 540, "teal", 46, en="b"))

# ---- 2. Cómo funciona
v.b("¿Cómo funciona? Dos pasos. [[a]]Uno: en el espacio original, con todas las columnas, mide qué tan probable es que cada cliente sea vecino de cada otro. [[b]]Dos: pone los puntos en un plano, y los va moviendo de a poco, hasta que las vecindades del mapa se parezcan a las originales.",
    CL(), tarjetas([(560, "azul", "1 · medir vecindades", "en el espacio original, con todas las columnas", "a"),
                    (1360, "violeta", "2 · acomodar el mapa", "mover puntos en 2D, de a poco, hasta imitar esas vecindades", "b")], w=720, h=420),
    sec="2. Cómo funciona")
v.b("No es una fórmula que se resuelve de una vez. [[a]]Es prueba y ajuste, una vuelta tras otra. [[b]]Por eso tarda, y por eso el resultado cambia un poco entre corridas.",
    P("prueba y ajuste, vuelta tras vuelta", 960, 800, "ambar", 50, en="a"), P("tarda · cambia entre corridas", 960, 920, "rojo", 46, en="b"))

# ---- 3. PCA vs t-SNE con datos
v.b("Veámoslo con los datos de la clase: los mil ciento veinticinco clientes de entrenamiento, con las quince columnas estandarizadas. [[a]]Aquí está el mapa de PCA en dos dimensiones. En rojo, los que se fugaron.",
    CL(), mapa("PCA", "gm", t=0.1, alfa_fuga=0.0), T("PCA · 2 componentes", 670, 160, tam=40, c="azul", id="tm"),
    U("gm", **{"series.1.alfa": 1.0}, en="a", dur=0.6), T("muestra de 500 de los 1.125 clientes", 670, 925, tam=28, c="gris", w="sb", t=0.4),
    sec="3. PCA frente a t-SNE")
v.b("Ahora mira cómo se reacomodan los mismos clientes con t-SNE, perplexity treinta. [[a]]Cada punto viaja a su lugar en el nuevo mapa. [[b]]Se forman islas de vecinos.",
    U("tm", texto="t-SNE · perplexity 30", color="violeta", en="a", dur=0.4), morph("gm", "p30", en="a", dur=2.2),
    P("aparecen «islas» de vecinos", 1450, 400, "violeta", 44, en="b", t=0.6))
v.b("¿Qué mapa conserva mejor a los vecinos? Se mide con la trustworthiness: [[a]]qué tanto se conservan los diez vecinos más cercanos de cada cliente. Uno es perfecto. [[b]]PCA en dos dimensiones: cero coma setecientos noventa y nueve. [[c]]t-SNE: cero coma novecientos ochenta y dos.",
    T("trustworthiness (10 vecinos)", 1450, 520, tam=34, c="gris", w="sb", en="a"), T("1 = vecinos conservados", 1450, 580, tam=30, c="gris", w="sb", en="a", t=0.3),
    N(TW["PCA"], 1300, 700, "{:.3f}", 70, "azul", en="b", dur=0.7), T("PCA", 1300, 780, tam=34, c="azul", w="sb", en="b"),
    N(TW["p30"], 1620, 700, "{:.3f}", 70, "violeta", en="c", dur=0.7), T("t-SNE", 1620, 780, tam=34, c="violeta", w="sb", en="c"))
v.b("Fíjate en algo más: [[a]]los rojos no forman una isla propia. [[b]]t-SNE no usa la etiqueta. La fuga no es un grupo que el mapa pueda descubrir solo.",
    P("la fuga no forma una isla propia", 1450, 880, "rojo", 38, en="a", mw=560), P("t-SNE no usa la etiqueta", 1450, 980, "gris", 36, en="b"))

# ---- 4. Perplexity
v.b("El hiperparámetro clave de t-SNE es perplexity. [[a]]En lenguaje simple: a cuántos vecinos le presto atención. [[b]]Baja, grupos chicos y muy locales. Alta, una vista más panorámica.",
    CL(), P("perplexity ≈ a cuántos vecinos presto atención", 960, 280, "violeta", 50, en="a"),
    tarjetas([(560, "teal", "baja (5)", "grupos chicos y muy locales", "b"), (1360, "azul", "alta (80)", "vista más panorámica", "b")], y=600, w=640, h=320),
    sec="4. Perplexity")
v.b("Míralo en los datos. [[a]]Con perplexity cinco, el mapa se desmenuza en piezas pequeñas.",
    CL(), mapa("p30", "gq", t=0.1), T("t-SNE · perplexity 30", 670, 160, tam=40, c="violeta", id="tq"),
    U("tq", texto="t-SNE · perplexity 5", color="teal", en="a", dur=0.4), morph("gq", "p5", en="a", dur=2.0),
    P("perplexity 5: piezas locales", 1450, 400, "teal", 44, en="a", t=1.2))
v.b("[[a]]Con perplexity ochenta, los grupos se consolidan, y la vista es más global.",
    U("tq", texto="t-SNE · perplexity 80", color="azul", en="a", dur=0.4), morph("gq", "p80", en="a", dur=2.0),
    P("perplexity 80: vista global", 1450, 520, "azul", 44, en="a", t=1.2))
v.b("¿Cuál es mejor? Ninguno, sin más. [[a]]La trustworthiness es casi igual en los tres: 0.977, 0.982, y 0.979. [[b]]Se recomienda probar entre cinco y cincuenta, y la perplexity siempre debe ser menor que el número de observaciones.",
    *[[N(TW[k], x, 690, "{:.3f}", 52, c, en="a", t=0.2 * j, dur=0.6), T(lab, x, 760, tam=30, c=c, w="sb", en="a", t=0.2 * j)]
      for j, (k, x, c, lab) in enumerate((("p5", 1260, "teal", "perp. 5"), ("p30", 1450, "violeta", "perp. 30"), ("p80", 1640, "azul", "perp. 80")))],
    P("probar entre 5 y 50 · siempre < n", 1450, 880, "ambar", 38, en="b"))
v.b("En el caso aplicado, la pregunta era qué mapa mostrarle a la gerencia. [[a]]Para un mapa general de tipos de cliente, una perplexity alta, como ochenta. [[b]]Para buscar subgrupos pequeños y específicos, una baja, como cinco.",
    CL(), tarjetas([(560, "azul", "gerencia: mapa general", "perplexity alta (80)", "a"), (1360, "teal", "nichos y subgrupos", "perplexity baja (5)", "b")], w=720, h=360))

# ---- 5. Semillas y costo
v.b("Ahora, dos limitaciones. La primera: [[a]]t-SNE no es determinista. Con otra semilla, sale otro dibujo. [[b]]Mira: misma perplexity treinta, semilla cero, [[c]]y semilla uno.",
    CL(), P("no determinista", 1450, 330, "rojo", 52, en="a"), mapa("s0", "gs", t=0.1), T("perplexity 30 · semilla 0", 670, 160, tam=40, c="violeta", id="ts", en="b"),
    U("ts", texto="perplexity 30 · semilla 1", en="c", dur=0.4), morph("gs", "s1", en="c", dur=2.2),
    P("las islas cambian de lugar", 1450, 480, "ambar", 44, en="c", t=0.6), sec="5. Semillas, costo y transform")
v.b("La segunda: [[a]]es costoso. Compara todas las parejas de clientes. [[b]]Con mil ciento veinticinco clientes, son mil ciento veinticinco por mil ciento veinticuatro, dividido en dos: seiscientas treinta y dos mil doscientas cincuenta parejas.",
    CL(), P("costoso: compara todas las parejas", 960, 260, "rojo", 54, en="a"),
    T("1.125 · 1.124 / 2", 960, 450, tam=70, c="tinta", en="b"), T("= 632.250 parejas", 960, 590, tam=80, c="violeta", en="b", t=0.6))
v.b("Y la más importante para producción: [[a]]t-SNE no tiene transform. Solo tiene fit transform. [[b]]Si llega un cliente nuevo, no hay cómo ubicarlo en el mapa: habría que recalcular todo desde cero.",
    CL(), T("TSNE(...).fit_transform(X)", 960, 300, tam=56, c="tinta", w="m", en="a"), T(".transform(X_nuevo)", 960, 440, tam=56, c="gris", w="m", id="ttr", en="a", t=0.5),
    U("ttr", tachado=1.0, en="a", t=1.2, dur=0.6), P("cliente nuevo → recalcular todo", 960, 640, "rojo", 54, en="b"))

# ---- 6. Comparación y alternativas
FILAS = [("combina variables", "lineal", "no lineal"), ("conserva", "varianza global", "vecindades locales"), ("mismo resultado cada vez", "sí", "no"),
         ("velocidad", "rápido", "lento"), ("clientes nuevos (transform)", "sí", "no"), ("uso", "modelar y producir", "explorar y visualizar")]
v.b("Pongamos los dos lado a lado. [[a]]PCA combina de forma lineal, conserva la varianza global, da siempre lo mismo, es rápido, y transforma clientes nuevos. [[b]]t-SNE es no lineal, conserva vecindades, cambia entre corridas, es lento, y no transforma. [[c]]Para modelar, PCA. Para explorar, t-SNE.",
    CL(), T("PCA", 1060, 200, tam=48, c="azul"), T("t-SNE", 1540, 200, tam=48, c="violeta"),
    *[[T(f, 760, 300 + 95 * i, tam=32, c="gris", w="sb", anc="rm", en="a", t=0.12 * i), T(a, 1060, 300 + 95 * i, tam=36, c="azul", w="b", en="a", t=0.12 * i + 0.05),
       T(b, 1540, 300 + 95 * i, tam=36, c="violeta", w="b", en="b", t=0.12 * i)] for i, (f, a, b) in enumerate(FILAS)],
    R(1060, 300 + 95 * 5, 420, 80, "azul", 30, id="ru1", en="c", ent="fade"), R(1540, 300 + 95 * 5, 420, 80, "violeta", 30, id="ru2", en="c", ent="fade", t=0.3),
    sec="6. PCA frente a t-SNE, lado a lado")
v.b("Y hay más técnicas en la misma familia. [[a]]Lineales: análisis factorial, análisis discriminante y {SVD|ese ve de}. [[b]]No lineales, más complejas: autoencoders, Isomap, {local linear embedding|lócal línear embéding} y {UMAP|iú map}.",
    CL(), tarjetas([(560, "azul", "lineales", "análisis factorial · análisis discriminante · SVD", "a"),
                    (1360, "violeta", "no lineales", "autoencoders · Isomap · LLE · UMAP", "b")], w=720, h=380))

# ---- 7. Error típico
v.b("El error típico con t-SNE: [[a]]ver islas y concluir que hay clústeres. [[b]]t-SNE expande los grupos densos, y compacta los dispersos. El tamaño de las islas, y la distancia entre ellas, no se interpretan. [[c]]Y t-SNE no es K-Means: no hay centros.",
    CL(), T("«veo islas: hay clústeres»", 960, 300, tam=70, c="tinta", id="qis", en="a"), U("qis", tachado=1.0, en="b", dur=0.6),
    P("tamaño y distancia entre islas: no se interpretan", 960, 520, "rojo", 48, en="b", t=0.3), P("t-SNE no es K-Means: no hay centros", 960, 700, "ambar", 48, en="c"),
    sec="7. El error típico")

# ---- 8. Piensa
v.b("Piensa un segundo: [[a]]un colega queda impresionado con los mapas de t-SNE, y propone usarlo en vez de PCA, antes del modelo que va a producción. ¿Qué le dices?",
    CL(), T("«reemplacemos PCA por t-SNE en producción»", 960, 380, tam=56, c="tinta", en="a"), T("¿qué le dices?", 960, 560, tam=80, c="violeta", en="a", t=0.6),
    sec="8. Piensa un segundo", pausa=0.8)
v.b("Que no. [[a]]Primero, t-SNE no tiene transform: no puede ubicar clientes nuevos sin recalcular todo. [[b]]Segundo, no es determinista: el mismo cliente podría quedar en otro lugar. [[c]]Un modelo en producción necesita lo contrario.",
    CL(), P("no", 960, 240, "rojo", 80, ent="subir"), P("1 · sin transform: no ubica clientes nuevos", 960, 440, "rojo", 48, en="a"),
    P("2 · no determinista: cambia con la semilla", 960, 600, "ambar", 48, en="b"), P("producción = PCA en un Pipeline", 960, 780, "verde", 50, en="c"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]t-SNE conserva vecindades locales, y PCA la varianza global. [[b]]Perplexity es a cuántos vecinos presta atención; prueba entre cinco y cincuenta. [[c]]Las islas no prueban clústeres. [[d]]Y t-SNE es para explorar; para modelar y producir, PCA.",
    CL(), P("t-SNE: vecindades · PCA: varianza global", 960, 260, "violeta", 48, en="a"), P("perplexity: 5 a 50, siempre < n", 960, 420, "teal", 48, en="b"),
    P("las islas no prueban clústeres", 960, 580, "rojo", 48, en="c"), P("explorar: t-SNE · producir: PCA", 960, 740, "azul", 50, en="d"), sec="9. Para llevar")
v.b("Última parada antes del repaso: [[a]]el Pipeline sin fuga, y el caso aplicado con la base real de churn.",
    CL(), siguiente(9, "Pipeline sin fuga y el caso Churn real"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
