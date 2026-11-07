"""Genera el notebook guiado de la Clase 47 (Embeddings y representaciones vectoriales).

Uso normal:      python notebook_c47.py             -> Guias/notebooks/C47_Embeddings.ipynb
Modo de prueba:  C47_TEST=1 python notebook_c47.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c47_numeros.json (lo escribe figuras_c47.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib y scikit-learn (solo TfidfVectorizer). No necesita internet: los vectores de MiniLM vienen calculados en vectores.csv.
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C47_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C47_Embeddings_test.ipynb" if TEST else "C47_Embeddings.ipynb")
N = json.loads((HERE / "c47_numeros.json").read_text(encoding="utf-8"))
cells = []


def _get(obj, parts):
    if not parts:
        return obj
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if isinstance(obj, list):
            if i == 1 and parts[0].isdigit():
                return _get(obj[int(parts[0])], parts[1:])
        elif k in obj:
            return _get(obj[k], parts[i:])
    raise KeyError(parts)


def val(m):
    obj = _get(N, m.group(1).split("."))
    d = m.group(2)
    return str(obj) if d is None else f"{float(obj):.{int(d)}f}"


def fill(text):
    return re.sub(r"\[\[([\w.\-]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- encabezado
md("""
# Clase 47 · Embeddings y representaciones vectoriales
**Diplomado Data Science** · Prof. Eloy Alvarado · cómo convertir un texto en números para que un computador encuentre la ficha de ayuda que responde una pregunta (el caso ficticio *Aula Abierta*)

Material de apoyo:
- `Guias/C47_Guia_Embeddings.pdf` — guía visual de la clase.
- `Guias/apuntes/C47_Apuntes_Embeddings.pdf` — apuntes explicados paso a paso, con ejemplos resueltos.
- `Guias/apuntes/C47_Ejercicios_Cuaderno_Embeddings.pdf` — ejercicios en papel (antes o después de este notebook).
- Diapositivas y programa del profesor (carpeta «Clase 47 …»): `2026-10-27_Embeddings_Presentacion.pdf`, `embeddings.py`, `documentos.csv`, `consultas.csv`, `vectores.csv`.
- Transcripción: `VideosClases/Clase 47 … .txt` (versión corregida; la grabación empieza unos 5 minutos tarde y tiene un recreo).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib` y `scikit-learn`. Si no los tienes:
```
pip install numpy pandas matplotlib scikit-learn
```
**No necesitas internet ni descargar modelos**: los vectores de MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`, 384 coordenadas) ya vienen calculados en `vectores.csv`.

**Datos**: 12 fichas de ayuda (`documentos.csv`), 16 consultas de **validación** (12 con ficha correcta y 4 sin respuesta) y 16 de prueba en `consultas.csv`. Siguiendo al profesor, **la prueba se reserva y aquí no se usa**.

**Cómo leer los números**: las cifras de las diapositivas se reproducen **exactamente** con la receta del profesor (`embeddings.py`). Lo marcado como *nuestro cálculo* es propio.

### Objetivos
1. Convertir textos en **vectores de conteo** y ver por qué los sinónimos quedan perpendiculares.
2. Calcular **TF-IDF a mano** y comprobarlo con `TfidfVectorizer`.
3. Calcular **producto interno, norma y coseno** a mano y con numpy.
4. Usar los **embeddings precalculados** para armar la matriz de puntajes 16 × 12 y **ordenar** las fichas.
5. Evaluar con **rango, Hit@k y MRR**, y **elegir k**.
6. Tratar las **consultas sin respuesta** con un corte, y ver el caso de los **códigos A17/A71**.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 25 | A1 · Contar palabras, A2 · Geometría con numpy |
| B | 30 | B1 · TF-IDF a mano, B2 · Con scikit-learn, B3 · Un caso real (V11) |
| C | 25 | C1 · Cargar los embeddings, C2 · Una pareja y la matriz 16 × 12 |
| D | 35 | D1 · Rangos, D2 · Hit@k y MRR, D3 · Elegir k |
| E | 25 | E1 · Sin respuesta y cortes, E2 · A17 y A71, E3 · PCA |
| — | 25 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook.
""")

md("## Preparación")
code("""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
np.set_printoptions(suppress=True, precision=4)
print("scikit-learn:", sklearn.__version__)


def buscar_carpeta():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 47*/documentos.csv"):
            return p.parent
        if (base / "documentos.csv").exists():
            return base
    return None


CARPETA = buscar_carpeta()
assert CARPETA is not None, "No encuentro la carpeta de la Clase 47 (documentos.csv, consultas.csv, vectores.csv)."
documentos = pd.read_csv(CARPETA / "documentos.csv", encoding="utf-8-sig")
todas = pd.read_csv(CARPETA / "consultas.csv", encoding="utf-8-sig").fillna("")
consultas = todas[todas["conjunto"] == "validacion"].reset_index(drop=True)   # la prueba se reserva, como en el programa del profesor
ids_doc = documentos["id"].tolist()
ids_con = consultas["id"].tolist()
print("carpeta:", CARPETA.name)
print(len(documentos), "fichas;", len(consultas), "consultas de validación;", int((consultas["relevante"] != "").sum()), "con respuesta")
documentos[["id", "tema", "texto"]].head(4)
""")
md("""
### Mini-repaso
- **Vector**: lista ordenada de números; **producto interno** = multiplicar coordenada a coordenada y sumar; **norma** = raíz de la suma de cuadrados.
- **Coseno** = producto interno dividido por el producto de las normas: compara direcciones.
- **Embedding** (Clase 46, ahora con texto): vector que una red ya entrenada entrega para un texto.
- **Validación y prueba** (Clases 30 y 38): se elige con validación; la prueba se mira al final.
""")

# ================================================================== PARTE A
md("""
## Parte A · Texto → vector

### A1 · Contar palabras (diapositiva 7)
Elegimos un **vocabulario** y contamos cuántas veces aparece cada palabra. Todos los textos quedan con el mismo largo. Observa qué pasa con los sinónimos «clave» y «contraseña».
""")
code("""
vocab = ["clave", "contraseña", "video"]
textos = ["clave", "contraseña", "video clave"]
conteos = np.array([[t.split().count(w) for w in vocab] for t in textos], dtype=float)
print(pd.DataFrame(conteos.astype(int), index=[f"«{t}»" for t in textos], columns=vocab))


def coseno(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\\ncoseno(«clave», «contraseña») =", coseno(conteos[0], conteos[1]), "  <- sinónimos, pero para el conteo no se parecen")
print("coseno(«clave», «video clave») =", round(coseno(conteos[0], conteos[2]), 4), "  <- comparten la palabra «clave»")
""")
md("""
### A2 · Geometría en 2 dimensiones (diapositivas 15 a 24)
Contamos solo dos palabras: (veces «clave», veces «recuperar»). **q** = «recuperar clave», **A** = «clave clave», **B** = «recuperar clave» repetido 4 veces. Compara las tres medidas.
""")
code("""
q = np.array([1.0, 1.0])
A = np.array([2.0, 0.0])
B = np.array([4.0, 4.0])

for nombre, v in (("A", A), ("B", B)):
    print(f"{nombre}: distancia {np.linalg.norm(q - v):.3f} | producto interno {q @ v:.0f} | norma {np.linalg.norm(v):.4f} | coseno {coseno(q, v):.4f}")
print("norma de q:", round(float(np.linalg.norm(q)), 4))
print("producto interno con 2B =", q @ (2 * B), "(se duplica aunque la dirección no cambió)")
print("q normalizado:", q / np.linalg.norm(q), "| B normalizado:", B / np.linalg.norm(B))
""")
md("""
**Lectura** (coincide con las diapositivas 17 a 24): la menor **distancia** favorece a A ([[geom.dist_a|3]] frente a [[geom.dist_b|3]]); el mayor **coseno** favorece a B (1 frente a [[geom.cos_a|4]]) porque tiene la misma dirección que q. El producto interno mezcla dirección y tamaño. El coseno 0.8 no es 80 % de probabilidad.
""")
md("""
### ✍️ Ejercicio 1 · Vectores de conteo
Vocabulario `["clase", "grabación", "ver", "video"]`. Textos: a = «ver video», b = «ver grabación», c = «ver clase clase».

1. Construye los tres vectores de conteo.
2. Calcula el coseno de a con b y de a con c.
3. ¿Qué coseno tienen «video» y «grabación» por sí solos? ¿Qué dice eso de los conteos?
""")
code("""
# tu código aquí
""")
solution("""
voc = ["clase", "grabación", "ver", "video"]
tx = {"a": "ver video", "b": "ver grabación", "c": "ver clase clase"}
vv = {k: np.array([t.split().count(w) for w in voc], float) for k, t in tx.items()}
print(vv)
print("cos(a, b) =", round(coseno(vv["a"], vv["b"]), 4), "| cos(a, c) =", round(coseno(vv["a"], vv["c"]), 4))
print("cos(video, grabación) =", coseno([0, 0, 0, 1], [0, 1, 0, 0]))
""", "a = (0, 0, 1, 1), b = (0, 1, 1, 0), c = (2, 0, 1, 0). cos(a, b) = 0.5 (comparten «ver») y cos(a, c) = 1/√10 ≈ 0.3162. «video» y «grabación» tienen coseno 0: para un conteo no tienen nada en común aunque signifiquen lo mismo.")

# ================================================================== PARTE B
md("""
## Parte B · TF-IDF: contar dándole más peso a lo raro

### B1 · TF-IDF a mano con numpy (diapositivas 28 y 30)
Receta del profesor: pasar a minúsculas y sin tildes, separar en palabras (letras y dígitos), armar el **vocabulario con las 12 fichas**, contar (**TF**), medir en cuántas fichas aparece cada palabra (**DF**), calcular **IDF = ln((1 + N)/(1 + DF)) + 1**, multiplicar y **normalizar** cada fila.
""")
code("""
def separar_palabras(texto):
    texto = texto.lower().translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return re.findall("[a-z0-9]+", texto)


vocabulario = sorted({w for t in documentos["texto"] for w in separar_palabras(t)})
columna = {w: j for j, w in enumerate(vocabulario)}


def contar(textos):
    M = np.zeros((len(textos), len(vocabulario)))
    for i, t in enumerate(textos):
        for w in separar_palabras(t):
            if w in columna:          # las palabras fuera del vocabulario se ignoran
                M[i, columna[w]] += 1
    return M


def normalizar(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, n, out=np.zeros_like(M), where=n > 0)


C_doc = contar(documentos["texto"])
C_con = contar(consultas["texto"])
df = (C_doc > 0).sum(axis=0)                      # DF: en cuántas fichas aparece cada palabra
idf = np.log((1 + len(documentos)) / (1 + df)) + 1
print("palabras en el vocabulario:", len(vocabulario), "| palabras que están en una sola ficha:", int((df == 1).sum()))
for w in ("de", "el", "proyecto", "a17", "a71", "contrasena"):
    print(f"{w:11s} DF = {df[columna[w]]:2d}  IDF = {idf[columna[w]]:.4f}")
print("«clave» está en el vocabulario:", "clave" in columna)
""")
code("""
T_doc = normalizar(C_doc * idf)
T_con = normalizar(C_con * idf)
S_tfidf = T_con @ T_doc.T            # 16 consultas x 12 fichas
print("forma de S_tfidf:", S_tfidf.shape)

i = ids_con.index("V01")
print("V01 usa estas palabras del vocabulario:", [w for w in separar_palabras(consultas.loc[i, "texto"]) if w in columna])
print(pd.Series(S_tfidf[i], index=ids_doc).round(3).sort_values(ascending=False).head(6))
""")
md("""
**Lectura**: de las 7 palabras de V01 solo «a» está en el vocabulario, así que las fichas con puntaje distinto de 0 (D03, D09, D04, D12, D11) se activan **solo por una palabra de relleno**; D01 queda con puntaje 0 (diapositiva 34: lugar 6, puntaje 0). Con conteos no hay forma de unir «clave» con «contraseña».
""")
md("""
### B2 · Lo mismo con scikit-learn
`TfidfVectorizer` usa por defecto `smooth_idf=True` (IDF = ln((1 + N)/(1 + DF)) + 1) y normaliza con la norma L2: es **la misma receta**. Solo hay que pasarle la tokenización del profesor.
""")
code("""
vec = TfidfVectorizer(analyzer=separar_palabras)          # vocabulario y IDF se ajustan con las 12 fichas
Xd = vec.fit_transform(documentos["texto"])
Xq = vec.transform(consultas["texto"])
S_sk = (Xq @ Xd.T).toarray()
print("mismo vocabulario:", sorted(vec.vocabulary_) == vocabulario)
print("mismo IDF:", np.allclose(vec.idf_[[vec.vocabulary_[w] for w in vocabulario]], idf))
print("mismos puntajes 16 x 12:", np.allclose(S_sk, S_tfidf))
""")
md("""
### B3 · Un caso real: V11 «¿Cuándo vence el proyecto A17?» frente a D11
Solo hay palabras comunes en tres columnas. El coseno es la suma de esos tres productos (con los vectores ya normalizados).
""")
code("""
i, j = ids_con.index("V11"), ids_doc.index("D11")
comunes = [w for w in vocabulario if C_con[i, columna[w]] > 0 and C_doc[j, columna[w]] > 0]
tabla = pd.DataFrame({"DF": [df[columna[w]] for w in comunes],
                      "IDF": [idf[columna[w]] for w in comunes],
                      "peso en V11": [T_con[i, columna[w]] for w in comunes],
                      "peso en D11": [T_doc[j, columna[w]] for w in comunes]}, index=comunes)
tabla["producto"] = tabla["peso en V11"] * tabla["peso en D11"]
print(tabla.round(4))
print("suma de productos =", round(tabla["producto"].sum(), 4), "| puntaje del programa =", round(S_tfidf[i, j], 4))
""")
md("""
**Lectura**: «a17» (DF 1) aporta [[v11.comunes.0.prod|3]], «el» (DF 7) [[v11.comunes.1.prod|3]] y «proyecto» (DF 2) [[v11.comunes.2.prod|3]]; suma **[[v11.suma|3]]**. Las palabras raras pesan más, y por eso el código «a17» ayuda a TF-IDF.
""")
md("""
### ✍️ Ejercicio 2 · Tu propia consulta con TF-IDF
Escribe una consulta nueva (por ejemplo, «¿Cuándo es la entrega del proyecto A71?») y usa `vec.transform` para obtener sus puntajes con las 12 fichas. ¿Qué ficha queda primera? ¿Qué palabras de tu consulta están en el vocabulario? (Una consulta sin ninguna palabra del vocabulario da una fila de ceros: no hay ranking.)
""")
code("""
# tu código aquí
""")
solution("""
nueva = "¿Cuándo es la entrega del proyecto A71?"
xq = vec.transform([nueva])
puntajes = pd.Series((xq @ Xd.T).toarray()[0], index=ids_doc).sort_values(ascending=False)
print(puntajes.round(3).head(4))
print("palabras en el vocabulario:", [w for w in separar_palabras(nueva) if w in vec.vocabulary_])
""", "Queda primera D12 (la ficha de A71): «a71» está en una sola ficha y pesa mucho. Las palabras comunes como «el» o «de» aportan poco. Si escribes sinónimos que no están en las fichas (por ejemplo «clave»), el conteo no los ve.")

# ================================================================== PARTE C
md("""
## Parte C · Embeddings precalculados (MiniLM)

### C1 · Cargar los vectores (diapositivas 6, 14 y 31)
`vectores.csv` trae, para cada ficha y cada consulta, las 384 coordenadas (`e001` a `e384`) que entrega MiniLM. **No hay que entrenar ni descargar nada**: aquí solo hacemos inferencia ya calculada. Después de normalizar, cada vector mide 1.
""")
code("""
vect = pd.read_csv(CARPETA / "vectores.csv", encoding="utf-8-sig").set_index("id")
cols = [f"e{j:03d}" for j in range(1, 385)]
print("vectores totales:", vect.shape[0], "| coordenadas:", len(cols))
D = normalizar(vect.loc[ids_doc, cols].to_numpy())        # 12 x 384
Q = normalizar(vect.loc[ids_con, cols].to_numpy())        # 16 x 384
print("D:", D.shape, "| Q:", Q.shape)
print("norma de cada vector (todas 1):", np.allclose(np.linalg.norm(D, axis=1), 1), np.allclose(np.linalg.norm(Q, axis=1), 1))
""")
md("""
### C2 · Una pareja y la matriz 16 × 12 (diapositivas 32 y 33)
Para V01 «Olvidé mi clave…» y D01 (recuperar la contraseña): multiplicamos las 384 coordenadas y sumamos. Luego hacemos lo mismo con todas las parejas de una vez: **S = Q Dᵀ**.
""")
code("""
i, j = ids_con.index("V01"), ids_doc.index("D01")
productos = Q[i] * D[j]
print("tres primeros productos:", productos[:3].round(5))
print("suma de las otras 381:", round(float(productos[3:].sum()), 5))
print("suma de los 384 productos (coseno):", round(float(productos.sum()), 5))
print("positivos:", int((productos > 0).sum()), "| negativos:", int((productos < 0).sum()))

S_minilm = Q @ D.T                    # 16 x 12: una fila por consulta, una columna por ficha
print("\\nS_minilm:", S_minilm.shape, "| S[V01, D01] =", round(float(S_minilm[i, j]), 5))
pd.DataFrame(S_minilm, index=ids_con, columns=ids_doc).round(3).head(4)
""")
md("""
**Lectura** (diapositivas 32 y 33): los tres primeros productos son −0.00114, 0.00082 y −0.00218; las otras 381 coordenadas suman 0.45262, y el coseno es **[[v01.suma|5]]** ≈ 0.450. La fila V01 de la matriz: D01 0.450, D02 −0.150, D03 0.078 … D12 −0.128. Como los vectores están normalizados, el producto interno **es** el coseno.
""")
code("""
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (nombre, S) in zip(axs, (("TF-IDF", S_tfidf), ("MiniLM", S_minilm))):
    im = ax.imshow(S, cmap="BuPu", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(ids_doc, rotation=90, fontsize=8)
    ax.set_yticks(range(16))
    ax.set_yticklabels(ids_con, fontsize=8)
    for k, rel in enumerate(consultas["relevante"]):
        if rel:
            ax.add_patch(plt.Rectangle((ids_doc.index(rel) - 0.5, k - 0.5), 1, 1, fill=False, ec="green", lw=2))
    ax.set_title(f"Coseno {nombre} (verde = ficha correcta)", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
plt.show()
""")

# ================================================================== PARTE D
md("""
## Parte D · Ordenar y evaluar

### D1 · El rango de la ficha correcta (diapositivas 34 y 35)
Ordenamos cada fila de mayor a menor puntaje (empates por identificador de ficha, como el programa del profesor) y buscamos la posición **r** de la ficha relevante. Una fila nula (consulta sin palabras del vocabulario) no entrega ranking: aporta 0.
""")
code("""
def rango(S, i, relevante):
    if not relevante or np.linalg.norm(S[i]) == 0:
        return 0
    orden = sorted(range(len(ids_doc)), key=lambda j: (-S[i, j], ids_doc[j]))
    return orden.index(ids_doc.index(relevante)) + 1


resp = [i for i, r in enumerate(consultas["relevante"]) if r]          # consultas con ficha correcta (V01 a V12)
r_tf = [rango(S_tfidf, i, consultas.loc[i, "relevante"]) for i in resp]
r_ml = [rango(S_minilm, i, consultas.loc[i, "relevante"]) for i in resp]
tabla_r = pd.DataFrame({"correcta": consultas.loc[resp, "relevante"].values, "r TF-IDF": r_tf, "r MiniLM": r_ml}, index=[ids_con[i] for i in resp])
print(tabla_r.T)

i = ids_con.index("V02")
print("\\nV02 con MiniLM, primeros 3:", [(ids_doc[j], round(float(S_minilm[i, j]), 3)) for j in np.argsort(-S_minilm[i])[:3]])
""")
md("""
**Lectura**: V01 tiene r = 6 con TF-IDF y r = 1 con MiniLM (diapositiva 34); V02 queda con r = 2 en MiniLM: D06 ([[v02.scores.0|3]]), D02 ([[v02.scores.1|3]]), D01 ([[v02.scores.2|3]]) (diapositiva 35). La peor consulta de MiniLM es V04 (r = 5); la de TF-IDF, V07 (r = 11).
""")
md("""
### D2 · Hit@k y MRR (diapositivas 36 a 38)
- **Hit@k** = proporción de consultas con r ≤ k.
- **MRR** = promedio de 1/r (con r = 0 aporta 0).

Primero, el ejemplo de las diapositivas con V01, V02 y V04.
""")
code("""
def hit_en_k(rangos, k):
    return float(np.mean([1 if 0 < r <= k else 0 for r in rangos]))


def mrr(rangos):
    return float(np.mean([1 / r if r > 0 else 0.0 for r in rangos]))


tres = [r_ml[ids_con.index(x)] for x in ("V01", "V02", "V04")]
print("rangos de V01, V02, V04 con MiniLM:", tres)
print("Hit@3 =", round(hit_en_k(tres, 3), 3), "| MRR =", round(mrr(tres), 3))

print("\\nLas 12 consultas con respuesta:")
for nombre, rs in (("TF-IDF", r_tf), ("MiniLM", r_ml)):
    print(f"{nombre}: Hit@1 = {sum(1 for r in rs if 0 < r <= 1)}/12 | Hit@3 = {sum(1 for r in rs if 0 < r <= 3)}/12 ({hit_en_k(rs, 3):.3f}) | MRR = {mrr(rs):.3f}")
""")
md("""
**Lectura** (diapositivas 36 a 38 y 44): con V01, V02 y V04, Hit@3 = 2/3 ≈ 0.667 y MRR = (1 + 0.5 + 0.2)/3 ≈ 0.567. Con las 12 consultas: Hit@3 = 7/12 (TF-IDF) y 11/12 (MiniLM); Hit@1 = 5/12 y 10/12; MRR = [[hit.mrr_tfidf|3]] y [[hit.mrr_minilm|3]]. **Cambiar k no cambia el MRR**: usa el ranking completo.
""")
md("""
### D3 · Elegir k (diapositiva 44)
Mostrar más fichas aumenta los aciertos, pero también lo que hay que leer. Recorremos todos los k.
""")
code("""
ks = range(1, 13)
hits = pd.DataFrame({"k": list(ks),
                     "Hit@k TF-IDF": [hit_en_k(r_tf, k) for k in ks],
                     "Hit@k MiniLM": [hit_en_k(r_ml, k) for k in ks]}).set_index("k")
print(hits.round(3).T)
print("\\nk mínimo para recuperar las 12: TF-IDF =", next(k for k in ks if hit_en_k(r_tf, k) == 1), "| MiniLM =", next(k for k in ks if hit_en_k(r_ml, k) == 1))

ax = hits.plot(marker="o", figsize=(6.5, 3.2), color=["#D97706", "#0E7490"])
ax.axvline(3, color="gray", ls="--")
ax.set_ylabel("Hit@k (12 consultas con respuesta)")
ax.set_xticks(list(ks))
plt.tight_layout()
plt.show()
""")
md("""
**Lectura**: MiniLM llega a 12/12 recién con k = [[hit.k_cobertura_total_minilm]] (V04, r = 5); TF-IDF con k = [[hit.k_cobertura_total_tfidf]] (V07, r = 11). Son solo 12 consultas: una sola que cambie mueve Hit@k en 1/12 ≈ 0.083. **k se elige con validación**; la prueba se reserva.
""")
md("""
### ✍️ Ejercicio 3 · Qué consultas se pierden con k = 3
Para cada receta, lista las consultas con respuesta cuya ficha correcta **no** está entre las 3 primeras, con su rango. ¿Qué k haría falta para recuperar V04 con MiniLM?
""")
code("""
# tu código aquí
""")
solution("""
for nombre, rs in (("TF-IDF", r_tf), ("MiniLM", r_ml)):
    perdidas = [(ids_con[i], r) for i, r in zip(resp, rs) if not 0 < r <= 3]
    print(nombre, "pierde con k = 3:", perdidas)
print("k necesario para V04 con MiniLM:", r_ml[ids_con.index("V04")])
""", "TF-IDF pierde V01 (6), V02 (10), V03 (6), V04 (4) y V07 (11): 5 de 12. MiniLM pierde solo V04 (5). Para recuperar V04 con MiniLM hace falta k = 5.")

# ================================================================== PARTE E
md("""
## Parte E · Límites

### E1 · Consultas sin respuesta y cortes de aceptación (diapositivas 41 a 43)
V13 a V16 no tienen ficha correcta («¿Hay estacionamiento gratuito?»), pero el ranking igual entrega un primer lugar. Una idea: **aceptar** la consulta solo si su coseno máximo alcanza un **corte**.
""")
code("""
maximo = S_minilm.max(axis=1)
tiene = np.array([bool(r) for r in consultas["relevante"]])
print("V14 «precio total en pesos»: primera ficha =", ids_doc[int(S_minilm[ids_con.index('V14')].argmax())], "con", round(float(maximo[ids_con.index('V14')]), 3))
filas = []
for corte in (0.3, 0.5, 0.7, 0.9):
    acepta = maximo >= corte
    filas.append({"corte": corte, "con respuesta aceptadas": f"{int((tiene & acepta).sum())}/{int(tiene.sum())}", "sin respuesta aceptadas": f"{int((~tiene & acepta).sum())}/{int((~tiene).sum())}"})
print(pd.DataFrame(filas).set_index("corte"))
print("\\ncoseno máximo de las sin respuesta:", {q_: round(float(m_), 3) for q_, m_ in zip(np.array(ids_con)[~tiene], maximo[~tiene])})
print("coseno máximo más bajo entre las con respuesta:", ids_con[int(np.where(tiene)[0][maximo[tiene].argmin()])], round(float(maximo[tiene].min()), 3))
""")
md("""
**Lectura** (diapositiva 42): con corte 0.3 se aceptan 9 de 12 consultas respondibles y 0 de 4 sin respuesta; con 0.5, 3 y 0; con 0.7 y 0.9, ninguna (programa). Cada corte **rechaza consultas que sí tenían respuesta** (V02 llega solo a [[umbral_extra.min_resp|3]]). V14 («precio total») recupera D05, que habla de pagos pero no informa el monto: **recuperar no es responder**. Los cortes son ilustrativos y 4 consultas sin respuesta son pocas.
""")
md("""
### ✍️ Ejercicio 4 · Un corte propio
Recorre cortes de 0.20 a 0.40 de 0.02 en 0.02. Para cada uno, cuenta cuántas consultas con respuesta y cuántas sin respuesta se aceptan. ¿Hay algún corte que acepte todas las respondibles sin aceptar ninguna sin respuesta?
""")
code("""
# tu código aquí
""")
solution("""
for corte in np.arange(0.20, 0.401, 0.02):
    acepta = maximo >= corte
    print(f"corte {corte:.2f}: con respuesta {int((tiene & acepta).sum()):2d}/12 | sin respuesta {int((~tiene & acepta).sum())}/4")
""", "No hay un corte perfecto: V02 (con respuesta) tiene coseno máximo 0.236, por debajo de V13, V14 y V15 (sin respuesta, 0.276, 0.266 y 0.252). Para aceptar a V02 hay que bajar a 0.236, y entonces se aceptan 3 de las 4 sin respuesta. Un corte alto protege de responder lo que no está, pero rechaza consultas respondibles.")
md("""
### E2 · El caso de los códigos A17 y A71 (diapositiva 40)
D11 y D12 comparten casi todas las palabras: cambian el código y la fecha. Confundirlas entrega una fecha equivocada.
""")
code("""
j11, j12 = ids_doc.index("D11"), ids_doc.index("D12")
print("palabras distintas:", sorted(set(separar_palabras(documentos.loc[j11, "texto"])) - set(separar_palabras(documentos.loc[j12, "texto"]))),
      "vs", sorted(set(separar_palabras(documentos.loc[j12, "texto"])) - set(separar_palabras(documentos.loc[j11, "texto"]))))
for q_id, ok, otro in (("V11", "D11", "D12"), ("V12", "D12", "D11")):
    i = ids_con.index(q_id)
    print(f"{q_id}: TF-IDF {S_tfidf[i, ids_doc.index(ok)]:.3f} vs {S_tfidf[i, ids_doc.index(otro)]:.3f} | MiniLM {S_minilm[i, ids_doc.index(ok)]:.3f} vs {S_minilm[i, ids_doc.index(otro)]:.3f} "
          f"| rangos: TF-IDF {rango(S_tfidf, i, ok)}, MiniLM {rango(S_minilm, i, ok)}")
print("coseno entre las fichas D11 y D12: MiniLM", round(float(D[j11] @ D[j12]), 3), "| TF-IDF", round(float(T_doc[j11] @ T_doc[j12]), 3))
parejas = D @ D.T
np.fill_diagonal(parejas, -1)
mejor = np.unravel_index(parejas.argmax(), parejas.shape)
print("par de fichas más parecido en MiniLM:", ids_doc[mejor[0]], ids_doc[mejor[1]], round(float(parejas[mejor]), 3))
""")
md("""
**Lectura**: en V11 y V12 ambas recetas ponen la ficha correcta primero, pero el margen de MiniLM en V12 es pequeño ([[codigos.V12.minilm_rel|3]] frente a [[codigos.V12.minilm_otro|3]]), y D11 y D12 son el par de fichas más parecido de todos (coseno [[codigos.cos_D11_D12_minilm|3]]). «La utilidad de una representación incluye conservar diferencias decisivas para la tarea.» Con códigos, conviene verificar el dato exacto antes de responder.
""")
md("""
### E3 · Mirar los embeddings con PCA (diapositiva 45)
384 coordenadas no se pueden dibujar. El PCA de las 12 fichas centradas las resume en dos ejes; **el ranking sigue usando las 384**.
""")
code("""
Dc = D - D.mean(axis=0)
U, valores, Vt = np.linalg.svd(Dc, full_matrices=False)
prop = valores ** 2 / np.sum(valores ** 2)
print(f"PC1 = {100 * prop[0]:.1f} % | PC2 = {100 * prop[1]:.1f} % | juntos = {100 * prop[:2].sum():.1f} %")

pf = Dc @ Vt[:2].T
pq = (Q - D.mean(axis=0)) @ Vt[:2].T
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(pf[:, 0], pf[:, 1], s=60, color="#0E7490")
for k, nombre in enumerate(ids_doc):
    ax.annotate(nombre, pf[k], (pf[k, 0] + 0.01, pf[k, 1] + 0.015), color="#0E7490", fontweight="bold")
ax.scatter(pq[:, 0], pq[:, 1], marker="x", color="#D97706", s=25, label="consultas (proyectadas)")
ax.set_xlabel(f"PC1 ({100 * prop[0]:.1f} %)")
ax.set_ylabel(f"PC2 ({100 * prop[1]:.1f} %)")
ax.legend()
plt.tight_layout()
plt.show()
""")
md("""
**Lectura** (diapositiva 45): PC1 [[pca.prop.0|3]] y PC2 [[pca.prop.1|3]] (20.0 % y 14.2 %), juntos 34.3 %: el dibujo omite casi dos tercios de la variación. D11 y D12 aparecen casi juntas. El plano ayuda a explorar; la calidad de la búsqueda se juzga con consultas y relevancia.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (sin ayuda)

### ✍️ Evaluar una receta completa
Escribe una función `evaluar(S, k)` que reciba una matriz de puntajes `S` (16 × 12) y un `k`, y devuelva un diccionario con `hit_k`, `mrr` y `n` (consultas con respuesta). Úsala con `S_tfidf` y `S_minilm` y k = 1, 3 y 5. Responde:
1. ¿Cuál receta prefieres con k = 1? ¿Y con k = 5?
2. ¿Qué métrica no cambia al variar k?
3. ¿Qué te falta para decir que la receta «responde bien» y no solo «recupera bien»?
4. ¿Por qué no usamos las consultas de prueba para elegir k?
""")
code("""
# tu código aquí
""")
solution("""
def evaluar(S, k):
    rs = [rango(S, i, consultas.loc[i, "relevante"]) for i in resp]
    return {"hit_k": hit_en_k(rs, k), "mrr": mrr(rs), "n": len(rs)}


for nombre, S in (("TF-IDF", S_tfidf), ("MiniLM", S_minilm)):
    for k in (1, 3, 5):
        e = evaluar(S, k)
        print(f"{nombre:7s} k = {k}: Hit@k = {e['hit_k']:.3f} | MRR = {e['mrr']:.3f} | n = {e['n']}")
""", "Hit@1 = 0.417 (TF-IDF) y 0.833 (MiniLM); Hit@3 = 0.583 y 0.917; Hit@5 = 0.667 y 1.000; MRR = 0.551 y 0.892 en todos los k. (1) MiniLM en ambos casos. (2) El MRR. (3) Revisar el contenido de las fichas recuperadas (¿tiene la información suficiente, como en el precio total?) y evaluar la aceptación de consultas sin respuesta. (4) Porque la prueba debe reservarse para evaluar la receta final sin haberla elegido con esos resultados.")

md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué «clave» y «contraseña» quedan perpendiculares en un vector de conteo?
2. ¿Qué es el IDF y qué valor tiene una palabra que aparece en 1 de las 12 fichas?
3. ¿Qué diferencia hay entre distancia, producto interno y coseno? ¿Cuál no depende del largo?
4. ¿Por qué con vectores normalizados el coseno es solo la suma de productos?
5. ¿Qué forma tiene S = Q Dᵀ y qué significa cada casilla?
6. ¿Qué es el rango r? Con rangos 1, 2 y 5: ¿Hit@3 y MRR?
7. ¿Qué métrica no cambia con k y por qué?
8. ¿Por qué un ranking siempre tiene primer lugar aunque ninguna ficha responda?
9. ¿Qué costo tiene elegir un corte de aceptación?
10. ¿Por qué A17 y A71 son un riesgo, y qué revisarías?
11. ¿Qué porcentaje de la variación muestra el PCA de las fichas y para qué sirve el gráfico?

<details><summary><b>Respuestas</b></summary>

1. Cada palabra activa una columna distinta: (1, 0, 0) y (0, 1, 0). Su producto interno es 0 y el coseno 0.
2. IDF = ln((1 + N)/(1 + DF)) + 1: da más peso a palabras presentes en pocas fichas. Con N = 12 y DF = 1: ln(13/2) + 1 = 2.8718.
3. La distancia compara los puntos finales; el producto interno mezcla coordenadas y escala; el coseno compara direcciones. El coseno no depende del largo.
4. Porque el coseno es el producto interno dividido por las dos normas, y con normas 1 solo queda el producto interno.
5. 16 × 12: una fila por consulta, una columna por ficha; cada casilla es el coseno de esa pareja.
6. r es la posición de la ficha relevante. Con 1, 2 y 5: Hit@3 = 2/3 ≈ 0.667 y MRR = (1 + 0.5 + 0.2)/3 ≈ 0.567.
7. El MRR: usa el ranking completo (1/r), que no cambia con k.
8. Porque el ranking solo ordena puntajes; siempre hay un máximo, aunque sea bajo o insuficiente.
9. Un corte alto rechaza consultas con respuesta; uno bajo acepta consultas sin respuesta. No hay un corte perfecto en estos datos.
10. D11 y D12 son casi idénticas (coseno 0.891 en MiniLM): confundirlas entrega una fecha incorrecta. Revisaría el código exacto y el margen entre las dos primeras fichas.
11. 20.0 % + 14.2 % = 34.3 %: sirve para explorar; el ranking usa las 384 coordenadas.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Para seguir**: el programa `embeddings.py` hace lo mismo sin pandas y deja los resultados en CSV; en la próxima sesión el profesor anuncia el paso a algo generativo (final de la grabación).
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
