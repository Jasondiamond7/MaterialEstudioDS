import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, alerta, fila)

OUT = out("guion_c38_07_ranking.json")
v = V("Leer el ranking, la curva y los pliegues")
RK = D["h_ranking"]
RD = {r["modelo"]: r for r in RK}
F01 = [f["rmse"] for f in D["h_fold"]["Q_A0.1"]]
F1 = [f["rmse"] for f in D["h_fold"]["Q_A1"]]
DIF = [round(a, 3) - round(b, 3) for a, b in zip(F01, F1)]
NOM = {"Q_A0.1": "cuadrática, α = 0.1", "Q_A1": "cuadrática, α = 1", "Q_A10": "cuadrática, α = 10", "Q_A100": "cuadrática, α = 100",
       "LINEAL": "regresión lineal", "Q_A1000": "cuadrática, α = 1000", "MEDIA": "media de entrenamiento"}

v.portada("Leer el ranking", "La curva de validación y las diferencias por pliegue",
          "Séptima parada: los resultados de la validación cruzada del hormigón. Cómo leer el ranking sin creerle más de lo que dice.")

# ---- 1. Hook
v.b("La validación cruzada ya tiene un ganador: la cuadrática con alfa igual a 0.1. [[a]]Su RMSE fuera de pliegue es 8.229 megapascales. [[b]]El segundo lugar, alfa igual a 1, tiene 8.308. [[c]]La diferencia es 0.079 megapascales. ¿Eso prueba que alfa 0.1 es mejor? Esa es la pregunta de esta parada.",
    CL(), T("el ganador, por poco", 960, 150, tam=56, c="tinta", ent="subir"),
    T("α = 0.1", 560, 320, tam=46, c="verde", w="sb"), N(RD["Q_A0.1"]["rmse"], 560, 420, "{:.3f}", 96, "verde", en="a"),
    T("α = 1", 1360, 320, tam=46, c="azul", w="sb", en="b"), N(RD["Q_A1"]["rmse"], 1360, 420, "{:.3f}", 96, "azul", en="b"),
    P("diferencia: 0.079 MPa", 960, 620, "ambar", 54, en="c"), T("¿prueba superioridad?", 960, 750, tam=46, c="tinta", w="sb", en="c", t=1.5),
    real(1050, en="a"), sec="1. Ganar por poco")

# ---- 2. Analogía
v.b("Piensa en una carrera de cien metros que se define por foto final. [[a]]Un corredor gana por una centésima de segundo, y se lleva la medalla. [[b]]La regla de la carrera es clara: el primero que cruza, gana. [[c]]Pero nadie diría que es claramente más rápido: otro día, con otro viento, el orden podría invertirse. Elegir no es lo mismo que probar superioridad.",
    CL(), T("la foto final", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "la regla decide", "gana por una centésima", "a"), (1360, "gris", "no prueba", "otro día, el orden podría invertirse", "c")],
             y=450, w=640, h=300, tam_t=52, tam_d=40),
    P("elegir ≠ probar superioridad", 960, 760, "violeta", 50, en="c", t=1.5), ilustrativo(1000, texto="analogía"), sec="2. Analogía: la foto final")

# ---- 3. El ranking
Y0, PASO = 270, 92
filas_rk = []
for i, r in enumerate(RK):
    y = Y0 + PASO * i
    k = "a" if i < 4 else "b"
    col = "verde" if i == 0 else ("gris" if r["modelo"] == "MEDIA" else "azul")
    filas_rk += [T(NOM[r["modelo"]], 700, y, tam=32, c="tinta", w="sb", anc="rm", en=k, t=0.12 * i),
                 Bx(730, y - 28, 730 + r["rmse"] * 44, y + 28, col, None, 0, 10, grow="x", en=k, t=0.12 * i),
                 T(f"{r['rmse']:.3f}", 750 + r["rmse"] * 44, y, tam=32, c=col, w="b", anc="lm", en=k, t=0.12 * i + 0.4),
                 T(f"{r['mae']:.3f}", 1800, y, tam=30, c="gris", w="sb", anc="rm", en=k, t=0.12 * i + 0.4)]
v.b("Este es el ranking completo, como en la diapositiva veintiséis, con los números recalculados. [[a]]Arriba, las cuadráticas con alfa 0.1, alfa 1, alfa 10 y alfa 100. [[b]]Después, la regresión lineal, con 10.560; la cuadrática con alfa igual a mil, con 13.051; y al final, la media de entrenamiento, con 16.746. [[c]]A la derecha, en gris, el MAE fuera de pliegue: ordena a los siete candidatos exactamente igual.",
    CL(), T("RMSE fuera de pliegue (MPa) · diapositiva 26", 960, 150, tam=48, c="tinta", ent="subir"),
    T("MAE", 1800, 215, tam=28, c="gris", w="sb", anc="rm", en="c"), filas_rk,
    R(1760, Y0 + PASO * 3, 120, 620, "gris", r=20, en="c"), real(1050, en="a"), sec="3. El ranking")
v.b("¿Cómo se lee? [[a]]El menor RMSE fuera de pliegue elige al candidato: gana alfa igual a 0.1. [[b]]La media de entrenamiento muestra cuánto aportan los predictores: pasar de 16.746 a 8.229 es bajar el error a menos de la mitad, un 50.9 por ciento menos. [[c]]Pero la ventaja del ganador sobre alfa igual a 1 es solo 0.079 megapascales. La diapositiva lo dice claro: elegir no prueba superioridad estable.",
    CL(), T("cómo leer el ranking", 960, 150, tam=54, c="tinta", ent="subir"),
    P("menor RMSE OOF → elige: α = 0.1", 960, 300, "verde", 46, en="a"),
    T("frente a la media: 1 − 8.229 / 16.746 =", 1120, 450, tam=42, c="tinta", w="sb", anc="rm", en="b"),
    N(50.9, 1150, 450, "{:.1f}", 60, "azul", anc="lm", suf=" %", en="b", t=1.0), T("menos error", 1150, 520, tam=32, c="gris", w="sb", anc="lm", en="b", t=1.4),
    P("ventaja sobre α = 1: solo 0.079 MPa → no prueba superioridad estable", 960, 690, "ambar", 40, en="c"),
    diapo(26, en="a"), adicional(1010, en="b", texto="el 50.9 % es un cálculo adicional"))

# ---- 4. La curva de validación
AL = [0.1, 1, 10, 100, 1000]
LX = [math.log10(a) for a in AL]
AJ = [RD[f"Q_A{a:g}"]["rmse_ajuste"] for a in AL]
OO = [RD[f"Q_A{a:g}"]["rmse"] for a in AL]
v.b("La diapositiva veintisiete dibuja la curva de validación de las cinco cuadráticas. [[a]]En el eje horizontal, alfa, en escala logarítmica: más penalización hacia la derecha. [[b]]En verde, el error de ajuste: el de las mismas filas usadas para estimar. [[c]]En naranja, el error fuera de pliegue. La naranja siempre queda arriba: esa brecha describe el optimismo de medir dentro de la muestra.",
    CL(), T("curva de validación · diapositiva 27", 960, 140, tam=50, c="tinta", ent="subir"),
    G(400, 240, 1400, 840, -1.3, 3.3, 6, 14, [dict(xs=LX, ys=AJ, color="verde", grosor=6, marcadores=9, revela=0.0),
                                             dict(xs=LX, ys=OO, color="ambar", grosor=6, marcadores=9, revela=0.0)],
      xticks=[], yticks=[6, 8, 10, 12, 14], yfmt="{:.0f}", ylabel="RMSE", id="gv", tam_txt=28),
    *[T(f"α = {a:g}", 400 + (lx + 1.3) / 4.6 * 1000, 880, tam=28, c="gris", w="sb", en="a") for a, lx in zip(AL, LX)],
    U("gv", **{"series.0.revela": 1.0}, en="b", dur=1.5), T("ajuste", 1440, 440, tam=36, c="verde", anc="lm", en="b", t=0.8),
    U("gv", **{"series.1.revela": 1.0}, en="c", dur=1.5), T("fuera de pliegue", 1440, 360, tam=36, c="ambar", anc="lm", en="c", t=0.8),
    real(1050, en="b"), sec="4. La curva de validación")
v.b("Tres valores para leerla. [[a]]Con alfa igual a 0.1: ajuste 6.958, fuera de pliegue 8.229. [[b]]Con alfa igual a 10: 7.130 y 8.489. [[c]]Y con alfa igual a mil, suben los dos: 12.511 y 13.051. La penalización restringe demasiado. Dos advertencias de la diapositiva: elegimos con el error fuera de pliegue, no con el de ajuste; y una curva en forma de U no está garantizada.",
    CL(), T("tres valores de la cuadrícula (RMSE, MPa)", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(290, [("α", 520), ("ajuste", 960), ("fuera de pliegue", 1400)], tams=[40] * 3, colores=["gris"] * 3),
    fila(400, [("0.1", 520), (f"{AJ[0]:.3f}", 960), (f"{OO[0]:.3f}", 1400)], tams=[46, 52, 52], colores=["tinta", "verde", "ambar"], en="a",
         fondo="verdeclaro", x0=340, x1=1580, h=96),
    fila(510, [("10", 520), (f"{AJ[2]:.3f}", 960), (f"{OO[2]:.3f}", 1400)], tams=[46, 52, 52], colores=["tinta", "verde", "ambar"], en="b",
         fondo="linea", x0=340, x1=1580, h=96),
    fila(620, [("1000", 520), (f"{AJ[4]:.3f}", 960), (f"{OO[4]:.3f}", 1400)], tams=[46, 52, 52], colores=["tinta", "verde", "ambar"], en="c",
         fondo="rojoclaro", x0=340, x1=1580, h=96),
    T("se elige con OOF · la forma de U no está garantizada", 960, 790, tam=40, c="tinta", w="sb", en="c", t=2.5),
    diapo(27, en="a"), real(1010, en="a"))

# ---- 5. Las diferencias por pliegue
v.b("Ahora miremos pliegue por pliegue, como en la diapositiva veintiocho. [[a]]Restamos el RMSE de alfa 0.1 menos el de alfa 1, en cada pliegue. Si la barra queda bajo cero, gana 0.1; si queda sobre cero, gana 1. [[b]]Pliegue uno: 7.110 menos 7.099, más 0.011. Pliegue dos: menos 0.399. Pliegue tres: menos 0.030. Pliegue cuatro: menos 0.081. [[c]]Y pliegue cinco: 8.058 menos 7.957, más 0.101.",
    CL(), T("α = 0.1 menos α = 1, por pliegue · diapositiva 28", 960, 140, tam=48, c="tinta", ent="subir"),
    L(300, 520, 1620, 520, "tinta", 4, en="a"), T("0", 280, 520, tam=30, c="gris", anc="rm", en="a"),
    T("gana α = 1 ↑", 1650, 400, tam=30, c="ambar", w="sb", anc="lm", en="a"), T("gana α = 0.1 ↓", 1650, 640, tam=30, c="azul", w="sb", anc="lm", en="a"),
    *[x for i, d in enumerate(DIF) for x in [
        (Bx(420 + 260 * i, 520 - d * 600, 580 + 260 * i, 520, "ambar", None, 0, 8, grow="-y" if False else "y", en="c" if i == 4 else "b", t=0.5 * i)
         if d > 0 else Bx(420 + 260 * i, 520, 580 + 260 * i, 520 - d * 600, "azul", None, 0, 8, grow="-y", en="b", t=0.5 * i)),
        T(f"{d:+.3f}", 500 + 260 * i, (520 - d * 600 - 30) if d > 0 else (520 - d * 600 + 34), tam=32, c="ambar" if d > 0 else "azul", w="b",
          en="c" if i == 4 else "b", t=0.5 * i + 0.3),
        T(f"pliegue {i + 1}", 500 + 260 * i, 920, tam=30, c="gris", w="sb", en="a")]],
    real(1050, en="b"), sec="5. Pliegue por pliegue")
v.b("Entonces: [[a]]alfa 0.1 gana en tres pliegues, el dos, el tres y el cuatro. [[b]]Y alfa 1 gana en dos: el uno y el cinco. [[c]]El ganador del ranking no gana siempre: la mayor parte de su ventaja viene del pliegue dos.",
    CL(), T("¿quién gana en cada pliegue?", 960, 150, tam=54, c="tinta", ent="subir"),
    P("α = 0.1 gana en los pliegues 2, 3 y 4", 960, 330, "azul", 50, en="a"),
    P("α = 1 gana en los pliegues 1 y 5", 960, 480, "ambar", 50, en="b"),
    T("la mayor parte de la ventaja viene del pliegue 2 (−0.399)", 960, 650, tam=42, c="tinta", w="sb", en="c"), diapo(28, en="a"))

# ---- 6. Alerta del material
v.b("Aquí la grabación tiene un desliz. [[a]]Se dijo: «el alfa 0.1 mejora en tres pliegues: en el segundo, en el tercero y en el cuarto, y en el primero y en el tercero gana alfa igual a 1». [[b]]La primera parte es correcta. Pero la segunda debe decir: en el primero y en el quinto. [[c]]Lo confirma la tabla de la diapositiva: 7.110 contra 7.099 en el pliegue uno, y 8.058 contra 7.957 en el pliegue cinco. En el tercero gana 0.1: 8.806 contra 8.836.",
    CL(), alerta(170), cita("…en el segundo, en el tercero y en el cuarto, y en el primero y en el tercero gana alfa igual a 1", 320, en="a", tam=40),
    etiqueta("grabación de la clase", 410, en="a", t=0.4),
    P("debe decir: en el primero y en el QUINTO", 960, 530, "rojo", 48, en="b"),
    fila(690, [("pliegue 1", 560), ("7.110 > 7.099 → gana α = 1", 1180)], tams=[36, 40], colores=["gris", "ambar"], en="c", x0=360, x1=1560),
    fila(780, [("pliegue 3", 560), ("8.806 < 8.836 → gana α = 0.1", 1180)], tams=[36, 40], colores=["gris", "azul"], en="c", t=0.8, x0=360, x1=1560),
    fila(870, [("pliegue 5", 560), ("8.058 > 7.957 → gana α = 1", 1180)], tams=[36, 40], colores=["gris", "ambar"], en="c", t=1.6, x0=360, x1=1560),
    diapo(28, en="c"), sec="6. Un desliz en la grabación")

# ---- 7. No son réplicas independientes
v.b("Y una advertencia de la diapositiva: [[a]]estas cinco diferencias no son cinco réplicas independientes, ni un intervalo de confianza. [[b]]¿Por qué? Porque los ajustes comparten datos: dos ajustes cualesquiera de la validación cruzada usan en común tres de los cinco pliegues. [[c]]Sirven para ver si la ventaja es estable, no para calcular una prueba estadística.",
    CL(), T("cinco diferencias ≠ cinco réplicas", 960, 150, tam=54, c="tinta", ent="subir"),
    P("no son réplicas independientes ni un intervalo de confianza", 960, 300, "rojo", 44, en="a"),
    *[Bc(560 + 110 * i, 470, 90, 70, "azulclaro" if i not in (0,) else "ambarclaro", "azul" if i else "ambar", 3, 12, en="b", t=0.06 * i) for i in range(5)],
    T("ajuste del pliegue 1", 340, 470, tam=28, c="gris", w="sb", anc="rm", en="b"),
    *[Bc(560 + 110 * i, 570, 90, 70, "azulclaro" if i != 1 else "ambarclaro", "azul" if i != 1 else "ambar", 3, 12, en="b", t=0.4 + 0.06 * i) for i in range(5)],
    T("ajuste del pliegue 2", 340, 570, tam=28, c="gris", w="sb", anc="rm", en="b", t=0.4),
    T("comparten 3 de 5 pliegues", 1400, 520, tam=38, c="azul", w="sb", en="b", t=1.0),
    P("sirven para ver estabilidad, no para una prueba estadística", 960, 740, "teal", 44, en="c"), diapo(28, en="a"), sec="7. No son réplicas")

# ---- 8. Qué decide la regla y qué no
v.b("La diapositiva veintinueve lo ordena en dos columnas. [[a]]Lo que decide la regla: alfa 0.1 tiene 8.229, alfa 1 tiene 8.308; la menor cifra elige al primero. [[b]]Lo que no establece: que una diferencia de unos 0.079 megapascales sea una ventaja estable en otros datos. Además, el ganador está en el borde de la cuadrícula: 0.1 es el alfa más pequeño probado. [[c]]Y si quisiéramos preferir el modelo más simple dentro de cierta tolerancia, esa regla también debía fijarse antes de comparar.",
    CL(), T("una regla de elección no prueba superioridad · diapositiva 29", 960, 140, tam=46, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "lo que decide", "8.229 < 8.308: la menor cifra elige α = 0.1", "a"),
              (1360, "rojo", "lo que no establece", "una ventaja estable · un óptimo global (está en el borde)", "b")],
             y=450, w=680, h=340, tam_t=48, tam_d=36),
    P("¿preferir lo simple con tolerancia? → fijarlo ANTES", 960, 780, "violeta", 44, en="c"), diapo(29, en="a"), sec="8. Qué decide la regla")

# ---- 9. El error típico
v.b("El error típico: [[a]]presentar el ganador como «el mejor modelo», con certeza, por una diferencia de 0.079. [[b]]O al revés: mirar los resultados, ver que alfa 1 está muy cerca, y cambiar la elección porque parece más simple. [[c]]Las dos cosas rompen el protocolo: la regla se aplica tal como se fijó, y la incertidumbre se comunica.",
    CL(), K(330, 250, "cruz", 70, en="a"), T("«el mejor modelo, sin duda» por 0.079", 410, 250, tam=46, c="rojo", anc="lm", en="a"),
    K(330, 400, "cruz", 70, en="b"), T("cambiar la elección después de mirar", 410, 400, tam=46, c="rojo", anc="lm", en="b"),
    P("aplicar la regla fijada y comunicar la incertidumbre", 960, 600, "teal", 46, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Comparas dos modelos, A menos B, en cinco pliegues, y obtienes: menos 0.2, más 0.1, menos 0.3, más 0.2 y menos 0.1. [[b]]¿En cuántos pliegues gana A? ¿Y cuál es la diferencia promedio?",
    CL(), T("A − B por pliegue:  −0.2,  +0.1,  −0.3,  +0.2,  −0.1", 960, 300, tam=52, c="tinta", w="b", en="a"),
    P("¿en cuántos gana A?", 640, 500, "azul", 48, en="b"), P("¿diferencia promedio?", 1300, 500, "violeta", 48, en="b", t=0.4),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("El modelo A gana donde la diferencia es negativa: [[a]]en tres pliegues. [[b]]La diferencia promedio: menos 0.2, más 0.1, menos 0.3, más 0.2, menos 0.1, suma menos 0.3; sobre cinco, [[c]]menos 0.06. El modelo A gana la regla, pero por poco, y no en todos los pliegues: igual que alfa 0.1 en el hormigón.",
    CL(), T("A gana en", 1000, 320, tam=50, c="tinta", w="sb", anc="rm"), N(3, 1030, 320, "{:.0f}", 70, "azul", anc="lm", suf=" pliegues", en="a"),
    T("promedio = −0.3 / 5 =", 1000, 470, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(-0.06, 1030, 470, "{:.2f}", 70, "violeta", anc="lm", en="c"),
    P("gana la regla, por poco, y no en todos los pliegues", 960, 660, "teal", 44, en="c", t=1.0), ilustrativo(1000))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el menor RMSE fuera de pliegue elige, y la media de entrenamiento dice cuánto aportan los predictores. [[b]]La curva de validación compara ajuste y validación; se elige con la validación. [[c]]Y una diferencia pequeña, que cambia de signo entre pliegues, decide la elección, pero no prueba superioridad.",
    CL(), P("menor RMSE OOF elige · la media mide el aporte", 960, 300, "verde", 46, en="a"),
    P("curva: ajuste vs. validación; se elige con validación", 960, 460, "ambar", 44, en="b"),
    P("elegir ≠ probar superioridad", 960, 620, "violeta", 50, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]reajustar el modelo elegido con todo el entrenamiento, evaluarlo una sola vez en la prueba, y mirar sus residuos.",
    CL(), siguiente(8, "Reajustar, evaluar una vez y diagnosticar"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
