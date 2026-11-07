"""Cátedra 47. Búsqueda de textos con TF-IDF y embeddings.

Extraer el ZIP y mantener juntos el programa y los tres CSV.
Requiere Python 3.10 o posterior y NumPy: python -m pip install numpy
Ejecutar con: python embeddings.py
En VS Code, abrir la carpeta extraída y ejecutar las celdas # %% en orden.
Los CSV de resultados se guardan en mis_resultados_python y se reemplazan al repetir.

El caso ficticio Aula Abierta tiene 12 fichas de ayuda y 16 consultas de
validación; 12 tienen respuesta. Usamos vectores ya calculados, sin conexión.
Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
Variante ONNX cuantizada; media de tokens válidos y 384 coordenadas normalizadas.
"""

# %% 00. Configuración y funciones para leer, guardar y normalizar
from pathlib import Path
import csv
import re
import numpy as np

CARPETA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
K = 3  # Cambiar a 1 y repetir la sección 05 para comparar Hit@1 con Hit@3.
EVALUAR_PRUEBA = False  # Reservar la prueba hasta elegir la configuración final.
SALIDA = CARPETA / "mis_resultados_python"
SALIDA.mkdir(exist_ok=True)


def leer(nombre):
    with (CARPETA / nombre).open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def guardar(nombre, filas):
    # La marca UTF-8 permite reconocer las tildes al abrir el CSV en Excel.
    with (SALIDA / nombre).open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def normalizar(matriz):
    # Cada fila es un vector. Se divide por su longitud; una fila nula queda nula.
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return np.divide(matriz, normas, out=np.zeros_like(matriz), where=normas > 0)


# %% 01. Ejemplos de destilación y geometría
# Error entre el vector del maestro y la salida del estudiante.
referencia_maestro = np.array([0.2, 0.8])
salida_estudiante = np.array([0.4, 0.6])
error_destilacion = np.mean((salida_estudiante - referencia_maestro) ** 2)
print(f"Error ilustrativo de destilación: {error_destilacion:.2f}")

# Geometría: coordenadas = [veces 'clave', veces 'recuperar'].
# Estos conteos sirven para estudiar la geometría; no provienen de MiniLM.
textos_ejemplo = {
    "q": "recuperar clave",
    "A": "clave clave",
    "B": " ".join(["recuperar clave"] * 4),
}


def contar_ejemplo(texto):
    palabras = texto.split()
    return np.array([palabras.count("clave"), palabras.count("recuperar")], dtype=float)


def coseno(vector_a, vector_b):
    producto = np.dot(vector_a, vector_b)
    norma_a = np.linalg.norm(vector_a)
    norma_b = np.linalg.norm(vector_b)
    if norma_a == 0 or norma_b == 0:
        raise ValueError("El coseno no está definido para un vector de norma cero.")
    return float(producto / (norma_a * norma_b))


consulta_ejemplo = contar_ejemplo(textos_ejemplo["q"])
geometria = []
for nombre, texto in textos_ejemplo.items():
    vector = contar_ejemplo(texto)
    geometria.append({
        "texto_id": nombre,
        "texto": texto,
        "conteo_clave": int(vector[0]),
        "conteo_recuperar": int(vector[1]),
        "norma": float(np.linalg.norm(vector)),
        "producto_q": float(np.dot(consulta_ejemplo, vector)),
        "distancia_q": float(np.linalg.norm(consulta_ejemplo - vector)),
        "coseno_q": coseno(consulta_ejemplo, vector),
    })
    print(f"{nombre}: vector {vector}, coseno con q = {geometria[-1]['coseno_q']:.6f}")
guardar("ejemplo_geometria.csv", geometria)

# %% 02. Leer las fichas y seleccionar las consultas de validación
documentos = leer("documentos.csv")
todas_las_consultas = leer("consultas.csv")
conjunto = "prueba" if EVALUAR_PRUEBA else "validacion"
consultas = [fila for fila in todas_las_consultas if fila["conjunto"] == conjunto]
ids_documentos = [documento["id"] for documento in documentos]
print(f"Datos: {len(documentos)} fichas y {len(consultas)} consultas de {conjunto}.")

# %% 03. TF-IDF: una referencia basada en coincidencias de palabras
# TF (Term Frequency): conteo del término dentro del texto.
# DF (Document Frequency): número de documentos que contienen el término.
# IDF (Inverse Document Frequency): peso inverso de esa presencia en el corpus.
# El vocabulario y los IDF se calculan con las 12 fichas.


def separar_palabras(texto):
    texto = texto.lower().translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return re.findall("[a-z0-9]+", texto)


vocabulario = set()
for documento in documentos:
    vocabulario.update(separar_palabras(documento["texto"]))
vocabulario = sorted(vocabulario)
columna_palabra = {palabra: j for j, palabra in enumerate(vocabulario)}


def contar_palabras(textos):
    conteos = np.zeros((len(textos), len(vocabulario)))
    for i, texto in enumerate(textos):
        for palabra in separar_palabras(texto):
            if palabra in columna_palabra:
                j = columna_palabra[palabra]
                conteos[i, j] += 1
    return conteos


conteos_documentos = contar_palabras([fila["texto"] for fila in documentos])
conteos_consultas = contar_palabras([fila["texto"] for fila in consultas])
frecuencia_documental = (conteos_documentos > 0).sum(axis=0)
idf = np.log((1 + len(documentos)) / (1 + frecuencia_documental)) + 1
tfidf_documentos = normalizar(conteos_documentos * idf)
tfidf_consultas = normalizar(conteos_consultas * idf)
# @ multiplica matrices. .T pone los documentos en columnas.
puntajes_tfidf = tfidf_consultas @ tfidf_documentos.T

# %% 04. Leer embeddings y calcular los cosenos entre consultas y fichas
filas_vectores = leer("vectores.csv")
vectores = {}
for fila in filas_vectores:
    coordenadas = [float(fila[f"e{j:03d}"]) for j in range(1, 385)]
    vectores[fila["id"]] = np.array(coordenadas)

D = normalizar(np.stack([vectores[fila["id"]] for fila in documentos]))
Q = normalizar(np.stack([vectores[fila["id"]] for fila in consultas]))
# D: 12 x 384. Q: 16 x 384. S: 16 x 12 (una casilla por pareja).
# Los vectores tienen norma uno: producto interno = coseno.
puntajes_minilm = Q @ D.T

tabla_puntajes = []
for i, consulta in enumerate(consultas):
    fila = {"consulta": consulta["id"]}
    for j, identificador in enumerate(ids_documentos):
        fila[identificador] = float(puntajes_minilm[i, j])
    tabla_puntajes.append(fila)
guardar("puntajes_por_consulta.csv", tabla_puntajes)

if conjunto == "validacion":
    i = next(i for i, fila in enumerate(consultas) if fila["id"] == "V01")
    j = ids_documentos.index("D01")
    productos = Q[i] * D[j]  # Multiplicación coordenada a coordenada.
    detalle = []
    for coordenada in range(384):
        detalle.append({
            "coordenada": coordenada + 1,
            "consulta": float(Q[i, coordenada]),
            "documento": float(D[j, coordenada]),
            "producto": float(productos[coordenada]),
        })
    guardar("detalle_V01_D01.csv", detalle)
    print(f"V01 con D01: suma de 384 productos = {productos.sum():.8f}")

# %% 05. Ordenar los documentos y evaluar en las 12 consultas respondibles
# Hit@K (Hit Rate at K): proporción con ficha relevante entre los primeros K.
# MRR (Mean Reciprocal Rank): promedio de 1/posición de la ficha relevante.
# MRR usa el ranking completo: cambiar K no cambia MRR.
assert 1 <= K <= len(documentos), "K debe estar entre 1 y 12."
ranking = []
resumen = []
for modelo, puntajes in [("TF-IDF", puntajes_tfidf), ("MiniLM", puntajes_minilm)]:
    aciertos = []
    rangos_reciprocos = []
    for i, consulta in enumerate(consultas):
        # Mayor puntaje primero; empates resueltos por identificador de ficha.
        orden = sorted(range(len(documentos)), key=lambda j: (-puntajes[i, j], ids_documentos[j]))
        relevante = consulta["relevante"]  # Vacío si ninguna ficha responde.
        posicion_relevante = 0  # 0 indica que no se recuperó una respuesta.
        if relevante and np.linalg.norm(puntajes[i]) > 0:
            columna_relevante = ids_documentos.index(relevante)
            posicion_relevante = orden.index(columna_relevante) + 1
        # Sin coincidencias léxicas, una fila nula aporta cero a ambas métricas.
        if relevante:
            aciertos.append(int(0 < posicion_relevante <= K))
            reciproco = 1 / posicion_relevante if posicion_relevante > 0 else 0.0
            rangos_reciprocos.append(reciproco)
        for posicion, j in enumerate(orden, start=1):
            ranking.append({
                "modelo": modelo, "consulta": consulta["id"],
                "posicion": posicion, "documento": ids_documentos[j],
                "similitud": float(puntajes[i, j]),
                "relevante": int(ids_documentos[j] == relevante),
            })
    resumen.append({
        "modelo": modelo, "conjunto": conjunto,
        "n_respondibles": len(aciertos), "k": K,
        "hit_k": float(np.mean(aciertos)), "mrr": float(np.mean(rangos_reciprocos)),
    })
guardar("ranking.csv", ranking)
guardar("resumen.csv", resumen)
for fila in resumen:
    print(f"{fila['modelo']}: Hit@{K} = {fila['hit_k']:.6f}; MRR = {fila['mrr']:.6f}")

# %% 06. Cortes de aceptación: evaluar también las preguntas sin respuesta
# El coseno no es una probabilidad. Estos cortes son ilustrativos.
tiene_respuesta = np.array([bool(fila["relevante"]) for fila in consultas])
mejor_puntaje = puntajes_minilm.max(axis=1)
umbrales = []
for corte in [0.3, 0.5, 0.7, 0.9]:
    acepta = mejor_puntaje >= corte
    umbrales.append({
        "corte": corte,
        "respondibles_aceptadas": int((tiene_respuesta & acepta).sum()),
        "respondibles_total": int(tiene_respuesta.sum()),
        "sin_respuesta_aceptadas": int((~tiene_respuesta & acepta).sum()),
        "sin_respuesta_total": int((~tiene_respuesta).sum()),
    })
guardar("umbrales.csv", umbrales)
print("Cortes de aceptación:")
for fila in umbrales:
    print(f"  {fila['corte']:.1f}: {fila['respondibles_aceptadas']}/{fila['respondibles_total']} "
          f"consultas con respuesta aceptadas; "
          f"{fila['sin_respuesta_aceptadas']}/{fila['sin_respuesta_total']} sin respuesta aceptadas.")

# %% 07. Reproducir la variación explicada por la proyección PCA
# PCA se calcula sobre las 12 fichas centradas, mediante valores singulares.
# La búsqueda anterior conserva las 384 coordenadas originales.
documentos_centrados = D - D.mean(axis=0)
valores_singulares = np.linalg.svd(documentos_centrados, compute_uv=False)
proporcion_variacion = valores_singulares**2 / np.sum(valores_singulares**2)
print(f"PCA: PC1 = {100 * proporcion_variacion[0]:.1f}%; "
      f"PC2 = {100 * proporcion_variacion[1]:.1f}%; "
      f"ambos = {100 * proporcion_variacion[:2].sum():.1f}%")
print("Resultados guardados en:", SALIDA)
# Al modificar o añadir textos, hay que volver a calcular sus embeddings.
# Editar los CSV de textos no actualiza los valores guardados en vectores.csv.
