"""Figuras de la Clase 47 (Embeddings y representaciones vectoriales · Prof. Eloy Alvarado).
Fuentes: diapositivas 2026-10-27_Embeddings_Presentacion.pdf (48), transcripción de la clase (la grabación empieza ~5 min tarde y tiene un recreo
entre 1:10:26 y 1:28:13) y el programa del profesor embeddings.py (carpeta de la Clase 47: documentos.csv, consultas.csv, vectores.csv).
Reproducción: se EJECUTA el programa del profesor sobre una copia temporal (solo numpy; los vectores de MiniLM ya vienen calculados) y se compara
con una reimplementación propia; las cifras salen idénticas a las de las diapositivas (tabla NUM["comparacion"]).
Lo demás (ejemplos de tres textos, TF-IDF a mano de V11, hits@k para todo k, PCA con consultas, cos entre D11 y D12) es cálculo propio y se rotula
«nuestro cálculo». El conjunto de PRUEBA (T01-T16) no se toca: el profesor lo reserva hasta fijar la receta.
Escribe c47_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c47.py"""
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from figuras import GRID, INK, MUTED, save

C47 = "#0E7490"
C47_BG = "#ECFEFF"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
GREEN = "#16A34A"
GRAY = "#94A3B8"
VIOLET = "#7C3AED"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA = next(BASE.glob("Clase 47*"))
W_FIG = 7.4
_C = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.6):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


# ------------------------------------------------------------------ receta del profesor (reimplementada) + comprobación contra su programa
def leer(nombre):
    with (CARPETA / nombre).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def separar(texto):
    texto = texto.lower().translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return re.findall("[a-z0-9]+", texto)


def normalizar(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


def coseno(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def posicion(P, i, rel, ids):
    """Rango (1 = primero) de la ficha relevante; 0 si la fila es nula. Empates por identificador, como el profesor."""
    if not rel or np.linalg.norm(P[i]) == 0:
        return 0
    orden = sorted(range(len(ids)), key=lambda j: (-P[i, j], ids[j]))
    return orden.index(ids.index(rel)) + 1


def calc():
    if _C:
        return _C
    docs = leer("documentos.csv")
    todas = leer("consultas.csv")
    cons = [r for r in todas if r["conjunto"] == "validacion"]
    ids = [d["id"] for d in docs]
    cid = [c["id"] for c in cons]
    voc = sorted({w for d in docs for w in separar(d["texto"])})
    col = {w: j for j, w in enumerate(voc)}

    def contar(textos):
        M = np.zeros((len(textos), len(voc)))
        for i, t in enumerate(textos):
            for w in separar(t):
                if w in col:
                    M[i, col[w]] += 1
        return M

    Cd = contar([d["texto"] for d in docs])
    Cq = contar([c["texto"] for c in cons])
    df = (Cd > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    Td = normalizar(Cd * idf)
    Tq = normalizar(Cq * idf)
    S_tf = Tq @ Td.T
    vec = {r["id"]: np.array([float(r[f"e{j:03d}"]) for j in range(1, 385)]) for r in leer("vectores.csv")}
    D = normalizar(np.stack([vec[i] for i in ids]))
    Q = normalizar(np.stack([vec[i] for i in cid]))
    S_ml = Q @ D.T
    _C.update(docs=docs, cons=cons, ids=ids, cid=cid, voc=voc, col=col, Cd=Cd, Cq=Cq, df=df, idf=idf, Td=Td, Tq=Tq, S_tf=S_tf, D=D, Q=Q, S_ml=S_ml, contar=contar)
    return _C


def ejecutar_programa_del_profesor():
    """Corre embeddings.py sobre una copia en una carpeta temporal (escribe mis_resultados_python al lado) y devuelve las salidas leídas."""
    tmp = Path(tempfile.mkdtemp(prefix="c47_prof_"))
    try:
        for n in ("embeddings.py", "documentos.csv", "consultas.csv", "vectores.csv"):
            shutil.copy(CARPETA / n, tmp / n)
        r = subprocess.run([sys.executable, "-X", "utf8", "embeddings.py"], cwd=tmp, capture_output=True, text=True, encoding="utf-8", check=True)
        sal = tmp / "mis_resultados_python"
        out = {"stdout": r.stdout}
        for n in ("resumen.csv", "ranking.csv", "umbrales.csv", "puntajes_por_consulta.csv", "ejemplo_geometria.csv", "detalle_V01_D01.csv"):
            with (sal / n).open(encoding="utf-8-sig", newline="") as f:
                out[n] = list(csv.DictReader(f))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ------------------------------------------------------------------ números
def numeros():
    c = calc()
    ids, cid, docs, cons = c["ids"], c["cid"], c["docs"], c["cons"]
    S_tf, S_ml, D, Q = c["S_tf"], c["S_ml"], c["D"], c["Q"]
    resp = [i for i, q in enumerate(cons) if q["relevante"]]
    sin = [i for i, q in enumerate(cons) if not q["relevante"]]
    NUM["datos"] = dict(n_doc=len(docs), n_val=len(cons), n_resp=len(resp), n_sin=len(sin), dim=int(D.shape[1]), vocab=len(c["voc"]),
                        n_prueba=sum(1 for r in leer("consultas.csv") if r["conjunto"] == "prueba"),
                        modelo="paraphrase-multilingual-MiniLM-L12-v2")
    NUM["docs"] = [dict(id=d["id"], tema=d["tema"], texto=d["texto"]) for d in docs]
    NUM["consultas"] = [dict(id=q["id"], texto=q["texto"], relevante=q["relevante"]) for q in cons]
    prof = ejecutar_programa_del_profesor()

    # ---- geometría de dos coordenadas (diapositivas 15 a 24)
    q, a, b = np.array([1.0, 1.0]), np.array([2.0, 0.0]), np.array([4.0, 4.0])
    geo = dict(q=q.tolist(), a=a.tolist(), b=b.tolist(),
               dist_a=float(np.linalg.norm(q - a)), dist_b=float(np.linalg.norm(q - b)),
               dot_a=float(q @ a), dot_b=float(q @ b), dot_2b=float(q @ (2 * b)),
               norm_q=float(np.linalg.norm(q)), norm_a=float(np.linalg.norm(a)), norm_b=float(np.linalg.norm(b)),
               unit_q=(q / np.linalg.norm(q)).tolist(), unit_a=(a / np.linalg.norm(a)).tolist(), unit_b=(b / np.linalg.norm(b)).tolist(),
               cos_a=coseno(q, a), cos_b=coseno(q, b), ang_a=float(np.degrees(np.arccos(coseno(q, a)))), ang_b=float(np.degrees(np.arccos(np.clip(coseno(q, b), -1, 1)))))
    NUM["geom"] = geo
    NUM["destilacion"] = dict(maestro=[0.2, 0.8], estudiante=[0.4, 0.6], mse=float(np.mean((np.array([0.4, 0.6]) - np.array([0.2, 0.8])) ** 2)))
    # ---- conteo ingenuo (diapositiva 7)
    pal = ["clave", "contraseña", "video"]
    ejem = {"clave": [1, 0, 0], "contraseña": [0, 1, 0], "video clave": [1, 0, 1]}
    for t, v in ejem.items():
        w = t.split()
        assert [w.count(p) for p in pal] == v
    NUM["conteo"] = dict(palabras=pal, textos={t: v for t, v in ejem.items()}, cos_clave_contra=0.0)
    # ---- IDF y DF
    voc, df, idf, col = c["voc"], c["df"], c["idf"], c["col"]
    sel = ["de", "el", "proyecto", "recuperar", "contrasena", "a17", "a71", "arancel", "clave"]
    NUM["idf"] = dict(N=len(docs), tabla={w: dict(df=int(df[col[w]]), idf=float(idf[col[w]])) for w in sel if w in col},
                      ausentes=[w for w in sel if w not in col],
                      df_max=int(df.max()), palabra_df_max=voc[int(df.argmax())], df_1=int((df == 1).sum()))
    # ---- V01
    i01 = cid.index("V01")
    j01 = ids.index("D01")
    prod = Q[i01] * D[j01]
    r_tf = [posicion(S_tf, i, cons[i]["relevante"], ids) for i in range(len(cons))]
    r_ml = [posicion(S_ml, i, cons[i]["relevante"], ids) for i in range(len(cons))]
    toks01 = separar(cons[i01]["texto"])
    NUM["v01"] = dict(texto=cons[i01]["texto"], tokens=toks01, en_vocab=[w for w in toks01 if w in col],
                      solapa_d01=sorted(set(toks01) & set(separar(docs[j01]["texto"]))),
                      tfidf=[float(x) for x in S_tf[i01]], minilm=[float(x) for x in S_ml[i01]],
                      rango_tfidf=r_tf[i01], rango_minilm=r_ml[i01],
                      q3=[float(x) for x in Q[i01][:3]], d3=[float(x) for x in D[j01][:3]], p3=[float(x) for x in prod[:3]],
                      suma_resto=float(prod[3:].sum()), suma=float(prod.sum()), coseno_largo=coseno(Q[i01], D[j01]),
                      acum=[float(x) for x in np.cumsum(prod)], n_pos=int((prod > 0).sum()), n_neg=int((prod < 0).sum()),
                      pos_sum=float(prod[prod > 0].sum()), neg_sum=float(prod[prod < 0].sum()))
    NUM["v01"]["norm_q"] = float(np.linalg.norm(Q[i01]))
    NUM["v01"]["norm_d"] = float(np.linalg.norm(D[j01]))
    # ---- puntajes y rangos
    NUM["puntajes"] = dict(cons=cid, docs=ids, tfidf=np.round(S_tf, 6).tolist(), minilm=np.round(S_ml, 6).tolist())
    NUM["rangos"] = dict(cons=[cid[i] for i in resp], relevante=[cons[i]["relevante"] for i in resp],
                         tfidf=[r_tf[i] for i in resp], minilm=[r_ml[i] for i in resp])
    # ---- hits@k para todo k y MRR
    def hit(rs, k):
        return sum(1 for r in rs if 0 < r <= k)
    rt, rm = NUM["rangos"]["tfidf"], NUM["rangos"]["minilm"]
    NUM["hit"] = dict(k=list(range(1, 13)), tfidf=[hit(rt, k) for k in range(1, 13)], minilm=[hit(rm, k) for k in range(1, 13)], n=len(resp),
                      mrr_tfidf=float(np.mean([1 / r if r else 0 for r in rt])), mrr_minilm=float(np.mean([1 / r if r else 0 for r in rm])))
    NUM["hit"]["k_cobertura_total_minilm"] = next(k for k in range(1, 13) if hit(rm, k) == len(resp))
    NUM["hit"]["k_cobertura_total_tfidf"] = next(k for k in range(1, 13) if hit(rt, k) == len(resp))
    # tres consultas de la diapositiva 36-37
    sub = [cid.index(x) for x in ("V01", "V02", "V04")]
    NUM["tres"] = dict(cons=["V01", "V02", "V04"], rangos=[r_ml[i] for i in sub], hit3=sum(1 for i in sub if 0 < r_ml[i] <= 3),
                       mrr=float(np.mean([1 / r_ml[i] for i in sub])))
    # ---- V02 (diapositiva 35)
    i02 = cid.index("V02")
    o02 = sorted(range(12), key=lambda j: (-S_ml[i02, j], ids[j]))
    NUM["v02"] = dict(top3=[ids[j] for j in o02[:3]], scores=[float(S_ml[i02, j]) for j in o02[:3]])
    # ---- top 3 de V01
    o_tf = sorted(range(12), key=lambda j: (-S_tf[i01, j], ids[j]))
    o_ml = sorted(range(12), key=lambda j: (-S_ml[i01, j], ids[j]))
    NUM["v01"]["top3_tfidf"] = [(ids[j], float(S_tf[i01, j])) for j in o_tf[:3]]
    NUM["v01"]["top3_minilm"] = [(ids[j], float(S_ml[i01, j])) for j in o_ml[:3]]
    NUM["v01"]["n_tfidf_no_nulos"] = int((S_tf[i01] > 0).sum())
    # ---- V11 / V12 y los códigos A17 / A71
    d = {}
    for qid, rel, otro in (("V11", "D11", "D12"), ("V12", "D12", "D11")):
        i = cid.index(qid)
        d[qid] = dict(rel=rel, otro=otro, tfidf_rel=float(S_tf[i, ids.index(rel)]), tfidf_otro=float(S_tf[i, ids.index(otro)]),
                      minilm_rel=float(S_ml[i, ids.index(rel)]), minilm_otro=float(S_ml[i, ids.index(otro)]),
                      rango_tfidf=r_tf[i], rango_minilm=r_ml[i])
    NUM["codigos"] = d
    jj = (ids.index("D11"), ids.index("D12"))
    NUM["codigos"]["cos_D11_D12_minilm"] = float(D[jj[0]] @ D[jj[1]])
    NUM["codigos"]["cos_D11_D12_tfidf"] = float(c["Td"][jj[0]] @ c["Td"][jj[1]])
    mejor_otro = {}
    cd = D @ D.T
    np.fill_diagonal(cd, -1)
    for j, i_ in enumerate(ids):
        k = int(cd[j].argmax())
        mejor_otro[i_] = (ids[k], float(cd[j, k]))
    NUM["codigos"]["vecino_mas_cercano_minilm"] = mejor_otro
    t11, t12 = separar(docs[jj[0]]["texto"]), separar(docs[jj[1]]["texto"])
    NUM["codigos"]["tokens_distintos"] = [sorted(set(t11) - set(t12)), sorted(set(t12) - set(t11))]
    # ---- TF-IDF a mano de V11 con D11
    i11 = cid.index("V11")
    w_q = c["Tq"][i11]
    w_d = c["Td"][jj[0]]
    comunes = [w for w in voc if c["Cq"][i11, col[w]] > 0 and c["Cd"][jj[0], col[w]] > 0]
    filas = [dict(palabra=w, tf_q=int(c["Cq"][i11, col[w]]), tf_d=int(c["Cd"][jj[0], col[w]]), df=int(df[col[w]]), idf=float(idf[col[w]]),
                  bruto_q=float(c["Cq"][i11, col[w]] * idf[col[w]]), bruto_d=float(c["Cd"][jj[0], col[w]] * idf[col[w]]),
                  norm_q=float(w_q[col[w]]), norm_d=float(w_d[col[w]]), prod=float(w_q[col[w]] * w_d[col[w]])) for w in comunes]
    NUM["v11"] = dict(texto=cons[i11]["texto"], tokens=separar(cons[i11]["texto"]), comunes=filas, suma=float(sum(f["prod"] for f in filas)),
                      coseno=float(S_tf[i11, jj[0]]), norma_bruta_q=float(np.linalg.norm(c["Cq"][i11] * idf)), norma_bruta_d=float(np.linalg.norm(c["Cd"][jj[0]] * idf)),
                      n_tokens_d=len(separar(docs[jj[0]]["texto"])), n_vocab_d=int((c["Cd"][jj[0]] > 0).sum()))
    # ---- puntaje máximo, sin respuesta y cortes
    mx = S_ml.max(1)
    NUM["maximos"] = dict(cons=cid, minilm=[float(x) for x in mx], tfidf=[float(x) for x in S_tf.max(1)], con_respuesta=[bool(q["relevante"]) for q in cons],
                          top_minilm=[ids[int(S_ml[i].argmax())] for i in range(len(cons))],
                          top_tfidf=[ids[int(S_tf[i].argmax())] for i in range(len(cons))])
    um = []
    tiene = np.array([bool(q["relevante"]) for q in cons])
    for corte in (0.3, 0.5, 0.7, 0.9):
        ac = mx >= corte
        um.append(dict(corte=corte, resp_ac=int((tiene & ac).sum()), sin_ac=int((~tiene & ac).sum())))
    NUM["umbral"] = um
    NUM["umbral_extra"] = dict(min_resp=float(mx[tiene].min()), max_sin=float(mx[~tiene].max()), q_min_resp=cid[int(np.where(tiene)[0][mx[tiene].argmin()])],
                               q_max_sin=cid[int(np.where(~tiene)[0][mx[~tiene].argmax()])],
                               ac_en_corte_025=dict(resp=int((tiene & (mx >= 0.25)).sum()), sin=int((~tiene & (mx >= 0.25)).sum())),
                               ac_en_corte_020=dict(resp=int((tiene & (mx >= 0.20)).sum()), sin=int((~tiene & (mx >= 0.20)).sum())))
    # ---- PCA de las fichas
    Dc = D - D.mean(0)
    U, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    prop = sv ** 2 / (sv ** 2).sum()
    pts = Dc @ Vt[:2].T
    qpts = (Q - D.mean(0)) @ Vt[:2].T
    NUM["pca"] = dict(prop=[float(x) for x in prop[:2]], suma=float(prop[:2].sum()), fichas={i: pts[j].tolist() for j, i in enumerate(ids)},
                      consultas={i: qpts[j].tolist() for j, i in enumerate(cid)})
    # ---- comparación con las diapositivas / programa
    res = {(r["modelo"]): r for r in prof["resumen.csv"]}
    um_p = prof["umbrales.csv"]
    comp = []

    def add(item, diap, rep, igual):
        comp.append(dict(item=item, diapositiva=diap, reproducido=rep, igual=bool(igual)))
    add("Error de destilación (maestro [0.2, 0.8], estudiante [0.4, 0.6])", "0.04", fmt(NUM["destilacion"]["mse"], 2), abs(NUM["destilacion"]["mse"] - 0.04) < 1e-12)
    add("Distancia ‖q − a‖ y ‖q − b‖", "1.414 y 4.243", f"{geo['dist_a']:.3f} y {geo['dist_b']:.3f}", round(geo["dist_a"], 3) == 1.414 and round(geo["dist_b"], 3) == 4.243)
    add("Producto interno qᵀa, qᵀb y qᵀ(2b)", "2, 8 y 16", f"{geo['dot_a']:g}, {geo['dot_b']:g} y {geo['dot_2b']:g}", (geo["dot_a"], geo["dot_b"], geo["dot_2b"]) == (2, 8, 16))
    add("Normas ‖q‖, ‖a‖, ‖b‖", "√2, 2, √32", f"{geo['norm_q']:.4f}, {geo['norm_a']:.4f}, {geo['norm_b']:.4f}", np.allclose([geo["norm_q"], geo["norm_a"], geo["norm_b"]], [2 ** 0.5, 2, 32 ** 0.5]))
    add("Coseno de q con A y con B (también en el programa)", "0.7071 y 1", f"{geo['cos_a']:.4f} y {geo['cos_b']:.0f}", round(geo["cos_a"], 4) == 0.7071 and abs(geo["cos_b"] - 1) < 1e-12)
    pr = {r["texto_id"]: r for r in prof["ejemplo_geometria.csv"]}
    assert round(float(pr["A"]["coseno_q"]), 4) == 0.7071 and abs(float(pr["B"]["coseno_q"]) - 1) < 1e-12
    v = NUM["v01"]
    add("V01 · D01: primeros tres productos", "−0.00114; 0.00082; −0.00218", "; ".join(f"{x:.5f}".replace("-", "−") for x in v["p3"]),
        [round(x, 5) for x in v["p3"]] == [-0.00114, 0.00082, -0.00218])
    add("V01 · D01: coordenadas 1 a 3 de V01 y de D01", "0.04347, 0.02170, 0.02971 y −0.02631, 0.03797, −0.07323",
        ", ".join(f"{x:.5f}" for x in v["q3"]) + " y " + ", ".join(f"{x:.5f}".replace("-", "−") for x in v["d3"]),
        [round(x, 5) for x in v["q3"]] == [0.04347, 0.02170, 0.02971] and [round(x, 5) for x in v["d3"]] == [-0.02631, 0.03797, -0.07323])
    add("V01 · D01: suma de las otras 381 coordenadas", "0.45262", f"{v['suma_resto']:.5f}", round(v["suma_resto"], 5) == 0.45262)
    add("V01 · D01: coseno (suma de 384 productos)", "≈ 0.45013 (0.450)", f"{v['suma']:.5f}; programa: {float(prof['stdout'].split('suma de 384 productos = ')[1].split()[0]):.5f}",
        round(v["suma"], 5) == 0.45013 and abs(v["suma"] - float(prof["stdout"].split("suma de 384 productos = ")[1].split()[0])) < 1e-8)
    mm = lambda qid, did: float(S_ml[cid.index(qid), ids.index(did)])
    add("Matriz 16 × 12 (V01: D01, D02, D03, D12)", "0.450; −0.150; 0.078; −0.128", "; ".join(f"{mm('V01', x):.3f}".replace("-", "−") for x in ("D01", "D02", "D03", "D12")),
        [round(mm("V01", x), 3) for x in ("D01", "D02", "D03", "D12")] == [0.450, -0.150, 0.078, -0.128])
    add("Matriz (V02 y V03 con D01, D02, D03, D12)", "V02: 0.131; 0.147; −0.022; 0.098 · V03: 0.052; 0.056; 0.366; 0.048 (la lámina alinea los dos últimos en la fila siguiente)",
        "V02: " + "; ".join(f"{mm('V02', x):.3f}".replace("-", "−") for x in ("D01", "D02", "D03", "D12")) + " · V03: " + "; ".join(f"{mm('V03', x):.3f}" for x in ("D01", "D02", "D03", "D12")),
        [round(mm("V02", x), 3) for x in ("D01", "D02", "D03", "D12")] == [0.131, 0.147, -0.022, 0.098] and [round(mm("V03", x), 3) for x in ("D01", "D02", "D03", "D12")] == [0.052, 0.056, 0.366, 0.048])
    add("V01 con TF-IDF: tres primeros (D03, D09, D04)", "0.162; 0.160; 0.156", "; ".join(f"{p:.3f}" for _, p in v["top3_tfidf"]) + " (" + ", ".join(i for i, _ in v["top3_tfidf"]) + ")",
        [i for i, _ in v["top3_tfidf"]] == ["D03", "D09", "D04"] and [round(p, 3) for _, p in v["top3_tfidf"]] == [0.162, 0.160, 0.156])
    add("V01 con TF-IDF: lugar de D01 y puntaje", "lugar 6; puntaje 0", f"lugar {v['rango_tfidf']}; puntaje {v['tfidf'][0]:g}", v["rango_tfidf"] == 6 and v["tfidf"][0] == 0)
    add("V01 con MiniLM: D01, D03, D05", "0.450; 0.078; 0.066", "; ".join(f"{p:.3f}" for _, p in v["top3_minilm"]) + " (" + ", ".join(i for i, _ in v["top3_minilm"]) + ")",
        [i for i, _ in v["top3_minilm"]] == ["D01", "D03", "D05"] and [round(p, 3) for _, p in v["top3_minilm"]] == [0.450, 0.078, 0.066])
    add("V02 con MiniLM: posiciones 1, 2 y 3", "D06; D02; D01", "; ".join(NUM["v02"]["top3"]), NUM["v02"]["top3"] == ["D06", "D02", "D01"])
    add("Rangos MiniLM de V01, V02, V04", "1; 2; 5", "; ".join(str(r) for r in NUM["tres"]["rangos"]), NUM["tres"]["rangos"] == [1, 2, 5])
    add("Hit@3 de V01, V02, V04", "2/3 ≈ 0.667", f"{NUM['tres']['hit3']}/3 = {NUM['tres']['hit3'] / 3:.3f}", NUM["tres"]["hit3"] == 2)
    add("MRR de V01, V02, V04", "≈ 0.567", f"{NUM['tres']['mrr']:.3f}", round(NUM["tres"]["mrr"], 3) == 0.567)
    H = NUM["hit"]
    add("Hit@3 en las 12 respondibles (TF-IDF; MiniLM)", "7/12 = 0.583; 11/12 = 0.917", f"{H['tfidf'][2]}/12 = {H['tfidf'][2] / 12:.3f}; {H['minilm'][2]}/12 = {H['minilm'][2] / 12:.3f}",
        (H["tfidf"][2], H["minilm"][2]) == (7, 11))
    add("MRR en las 12 respondibles (TF-IDF; MiniLM)", "0.551; 0.892", f"{H['mrr_tfidf']:.3f}; {H['mrr_minilm']:.3f}", (round(H["mrr_tfidf"], 3), round(H["mrr_minilm"], 3)) == (0.551, 0.892))
    add("Hit@1 (TF-IDF; MiniLM)", "5/12; 10/12", f"{H['tfidf'][0]}/12; {H['minilm'][0]}/12", (H["tfidf"][0], H["minilm"][0]) == (5, 10))
    rp = {(r["modelo"]): r for r in prof["resumen.csv"]}
    add("Programa (K = 3): Hit@3 y MRR de cada receta", "0.583333 / 0.550631; 0.916667 / 0.891667",
        f"{float(rp['TF-IDF']['hit_k']):.6f} / {float(rp['TF-IDF']['mrr']):.6f}; {float(rp['MiniLM']['hit_k']):.6f} / {float(rp['MiniLM']['mrr']):.6f}",
        abs(float(rp["TF-IDF"]["hit_k"]) - H["tfidf"][2] / 12) < 1e-12 and abs(float(rp["MiniLM"]["mrr"]) - H["mrr_minilm"]) < 1e-12 and abs(float(rp["TF-IDF"]["mrr"]) - H["mrr_tfidf"]) < 1e-12)
    cd_ = NUM["codigos"]
    add("A17 / A71 (V11, V12): rango de la ficha correcta", "1 en TF-IDF y 1 en MiniLM", f"V11: {cd_['V11']['rango_tfidf']} y {cd_['V11']['rango_minilm']}; V12: {cd_['V12']['rango_tfidf']} y {cd_['V12']['rango_minilm']}",
        (cd_["V11"]["rango_tfidf"], cd_["V11"]["rango_minilm"], cd_["V12"]["rango_tfidf"], cd_["V12"]["rango_minilm"]) == (1, 1, 1, 1))
    i14 = cid.index("V14")
    add("«Precio total» (V14): ficha en primer lugar en ambas recetas", "D05 en ambas", f"TF-IDF: {NUM['maximos']['top_tfidf'][i14]}; MiniLM: {NUM['maximos']['top_minilm'][i14]}",
        NUM["maximos"]["top_tfidf"][i14] == "D05" and NUM["maximos"]["top_minilm"][i14] == "D05")
    ref = {(float(r["corte"])): (int(r["respondibles_aceptadas"]), int(r["sin_respuesta_aceptadas"])) for r in um_p}
    add("Cortes 0.3 y 0.5: respondibles / sin respuesta aceptadas (el programa agrega 0.7 y 0.9)", "0.3: 9/12 y 0/4 · 0.5: 3/12 y 0/4",
        "; ".join(f"{k}: {a_}/12 y {b_}/4" for k, (a_, b_) in ref.items()), all(ref[u["corte"]] == (u["resp_ac"], u["sin_ac"]) for u in um) and ref[0.3] == (9, 0) and ref[0.5] == (3, 0))
    pp = NUM["pca"]
    add("PCA de las 12 fichas: PC1, PC2 y suma", "20.0 %; 14.2 %; 34.3 %", f"{pp['prop'][0] * 100:.1f} %; {pp['prop'][1] * 100:.1f} %; {pp['suma'] * 100:.1f} %",
        (round(pp["prop"][0] * 100, 1), round(pp["prop"][1] * 100, 1), round(pp["suma"] * 100, 1)) == (20.0, 14.2, 34.3))
    # comparación fina con el programa: todos los puntajes y todos los rangos
    pm = prof["puntajes_por_consulta.csv"]
    dif_p = max(abs(float(r[i_]) - S_ml[k, j]) for k, r in enumerate(pm) for j, i_ in enumerate(ids))
    rk = prof["ranking.csv"]
    ok_rk = True
    for r in rk:
        if r["relevante"] == "1":
            pos = int(r["posicion"])
            k = cid.index(r["consulta"])
            esp = r_tf[k] if r["modelo"] == "TF-IDF" else r_ml[k]
            ok_rk &= (pos == esp)
    add("Programa: 16 × 12 puntajes MiniLM y rangos de las 12 respondibles", "—", f"máxima diferencia {dif_p:.1e}; rangos {'idénticos' if ok_rk else 'DISTINTOS'}", dif_p < 1e-9 and ok_rk)
    NUM["comparacion"] = comp
    NUM["todo_igual"] = bool(all(r["igual"] for r in comp))
    NUM["no_reproducible"] = ["la forma exacta de los vectores 'ejemplo' de las diapositivas 31 (16 de 384 coordenadas, solo colores)", "el entrenamiento de MiniLM, que aquí se usa ya calculado (vectores.csv)", "el desempeño en las 16 consultas de prueba, que el profesor reserva (no se evaluó)"]


# ------------------------------------------------------------------ dibujo
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def flecha(ax, p, q, color=MUTED, lw=1.5, ms=12, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=4))


def caja(ax, x, y, w, h, t, col, fs=9, fc="white", bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0", fc=fc, ec=col, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc or INK, fontweight="bold" if bold else "normal", linespacing=1.3, zorder=3)


def f_pipeline():
    c = calc()
    v = NUM["v01"]
    fig, ax = canvas(W_FIG, 3.3)
    ax.text(1, 31.2, "Las cuatro etapas del buscador de Aula Abierta", fontsize=10.5, fontweight="bold", color=C47, va="center")
    xs = [1.0, 19.5, 38.0, 56.5]
    nombres = ["1 · Representar", "2 · Comparar", "3 · Ordenar", "4 · Evaluar"]
    preg = ["¿cómo convertir un\ntexto en un vector?", "¿qué significa que dos\nvectores se parezcan?", "¿qué fichas mostramos\nprimero?", "¿las fichas encontradas\nrealmente responden?"]
    ej = ["«Olvidé mi clave…»\n→ 384 números\n(MiniLM)",
          f"coseno(V01, D01)\n= {v['coseno_largo']:.3f}\nuna casilla por pareja",
          f"D01 {v['minilm'][0]:.3f}  1.º\nD03 {v['minilm'][2]:.3f}  2.º\nD05 {v['minilm'][4]:.3f}  3.º",
          "¿D01 está entre los\nprimeros k? ¿alcanza\npara responder?"]
    for x, n, p, e in zip(xs, nombres, preg, ej):
        caja(ax, x, 6, 15.6, 22.5, "", C47, fc=C47_BG)
        ax.text(x + 7.8, 26.2, n, ha="center", va="center", fontsize=9.6, fontweight="bold", color=C47)
        ax.text(x + 7.8, 20.2, p, ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.3)
        ax.plot([x + 1.5, x + 14.1], [15.0, 15.0], color=C47, lw=0.8, alpha=0.5)
        ax.text(x + 7.8, 10.2, e, ha="center", va="center", fontsize=8.4, color=MUTED, linespacing=1.35)
    for k in range(3):
        flecha(ax, (xs[k] + 15.9, 17.2), (xs[k + 1] - 0.3, 17.2), color=C47, lw=1.8, ms=14)
    ax.text(37, 1.8, "Cada etapa tiene su propio criterio: un buen puntaje (etapa 2) no garantiza una buena respuesta (etapa 4)", ha="center", fontsize=8.6, color=AMBER, fontweight="bold")
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(0, 33)
    save(fig, "c47_etapas.png")


def f_conteo():
    fig = plt.figure(figsize=(W_FIG, 3.0))
    ax = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.text(0.2, 7.5, "Contar palabras: un vector por texto", fontsize=9.8, fontweight="bold", color=C47, va="center")
    pal = NUM["conteo"]["palabras"]
    xs = [3.3, 5.5, 7.7]
    for x, p in zip(xs, pal):
        ax.add_patch(Rectangle((x - 1.05, 5.2), 2.1, 1.1, fc=C47, ec="white", lw=1.5))
        ax.text(x, 5.75, p, ha="center", va="center", fontsize=8.8, color="white", fontweight="bold")
    ax.text(1.0, 6.4, "texto", fontsize=8.2, color=MUTED, ha="left")
    cols = [BLUE, AMBER, GREEN]
    for r, (t, v) in enumerate(NUM["conteo"]["textos"].items()):
        y = 4.3 - r * 1.4
        ax.text(0.2, y + 0.45, f"«{t}»", fontsize=9, va="center", color=INK)
        for x, val in zip(xs, v):
            ax.add_patch(Rectangle((x - 1.05, y - 0.1), 2.1, 1.1, fc="white" if val == 0 else "#CFFAFE", ec=GRID, lw=1.2))
            ax.text(x, y + 0.45, str(val), ha="center", va="center", fontsize=11, fontweight="bold", color=MUTED if val == 0 else INK)
    ax.text(0.2, 0.2, "Cada columna es una palabra del vocabulario.", fontsize=8.4, color=MUTED, va="center")
    ax2 = fig.add_axes([0.58, 0.15, 0.38, 0.62])
    ax2.set_xlim(-0.25, 1.45)
    ax2.set_ylim(-0.25, 1.45)
    ax2.set_aspect("equal")
    ax2.annotate("", xy=(1.2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax2.annotate("", xy=(0, 1.2), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2))
    ax2.text(1.22, 0.04, "«clave»\n(1, 0)", color=BLUE, fontsize=8.6, ha="center", va="bottom", fontweight="bold")
    ax2.text(0.05, 1.22, "«contraseña» (0, 1)", color=AMBER, fontsize=8.6, ha="left", va="bottom", fontweight="bold")
    ax2.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=90, color=INK, lw=1.3))
    ax2.plot([0.17, 0.17, 0], [0, 0.17, 0.17], color=INK, lw=1)
    ax2.text(0.38, 0.38, "90°\ncoseno 0", fontsize=8.4, color=INK, ha="left", va="center")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    fig.text(0.58, 0.94, "Sinónimos = flechas perpendiculares", fontsize=9.8, fontweight="bold", color=C47, va="center")
    save(fig, "c47_conteo.png")


def f_sin_palabras():
    c = calc()
    v = NUM["v01"]
    fig, ax = canvas(W_FIG, 3.3)
    ax.text(1, 31.0, "V01 y D01 no comparten ninguna palabra: para un conteo no se parecen en nada", fontsize=10, fontweight="bold", color=C47, va="center")
    q_tok = v["tokens"]
    d_tok = sorted(set(separar(c["docs"][0]["texto"])), key=lambda w: separar(c["docs"][0]["texto"]).index(w))
    ax.text(1, 25.4, "Palabras de la consulta V01", fontsize=8.8, color=MUTED, va="center")
    x = 1
    for w in q_tok:
        wd = 1.9 + 0.95 * len(w)
        ax.add_patch(FancyBboxPatch((x, 20.2), wd, 3.4, boxstyle="round,pad=0.1,rounding_size=0.8", fc="#DBEAFE", ec=BLUE, lw=1.2))
        ax.text(x + wd / 2, 21.9, w, ha="center", va="center", fontsize=9, color=INK)
        x += wd + 1.2
    ax.text(1, 16.0, "Palabras de la ficha D01 (Acceso)", fontsize=8.8, color=MUTED, va="center")
    x, y = 1, 10.6
    for w in d_tok:
        wd = 1.9 + 0.95 * len(w)
        if x + wd > W_FIG * 10 - 1:
            x, y = 1, y - 4.4
        ax.add_patch(FancyBboxPatch((x, y), wd, 3.2, boxstyle="round,pad=0.1,rounding_size=0.8", fc="#FEF3C7", ec=AMBER, lw=1.0))
        ax.text(x + wd / 2, y + 1.6, w, ha="center", va="center", fontsize=8.2, color=INK)
        x += wd + 1.0
    ax.text(W_FIG * 10 - 1, 27.4, f"palabras en común: {len(v['solapa_d01'])}", ha="right", fontsize=9.4, color=RED, fontweight="bold")
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(0, 33)
    save(fig, "c47_sin_palabras.png")


def f_idf():
    c = calc()
    I = NUM["idf"]
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 3.1), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axs[0]
    dfs = np.arange(0, 13)
    ax.plot(dfs, np.log(13 / (1 + dfs)) + 1, color=C47, lw=2)
    pos = {"de": (-0.2, 0.0), "el": (0.6, 0.15), "proyecto": (0.9, 0.12), "recuperar": (0.9, -0.1), "a17": (0.9, 0.12)}
    for w, d, txt, off in (("de", I["tabla"]["de"], "«de»\nDF = 11, IDF = 1.08", (-3.0, 0.55)), ("proyecto", I["tabla"]["proyecto"], "«proyecto»\nDF = 2, IDF = 2.47", (3.2, 0.5)),
                           ("a17", I["tabla"]["a17"], "«a17» y «contrasena»\nDF = 1, IDF = 2.87", (3.4, 0.75))):
        ax.scatter([d["df"]], [d["idf"]], color=AMBER, zorder=4, s=36)
        ax.annotate(txt, (d["df"], d["idf"]), (d["df"] + off[0], d["idf"] + off[1]), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8), va="bottom")
    ax.set_xlabel("DF: fichas que contienen la palabra", fontsize=9)
    ax.set_ylabel("IDF = ln(13 / (1 + DF)) + 1", fontsize=9)
    ax.set_ylim(0.8, 4.3)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Cuanto más rara, más pesa", 9.6)
    style(ax)
    ax = axs[1]
    w = ["de", "el", "proyecto", "contrasena", "a17"]
    w = [x for x in w if x in I["tabla"]]
    vals = [I["tabla"][x]["idf"] for x in w]
    cols = [GRAY if I["tabla"][x]["df"] > 5 else C47 for x in w]
    ax.barh(range(len(w)), vals, color=cols)
    for k, x in enumerate(w):
        ax.text(vals[k] + 0.05, k, f"{vals[k]:.2f}  (DF {I['tabla'][x]['df']})", va="center", fontsize=8.6, color=INK)
    ax.set_yticks(range(len(w)))
    ax.set_yticklabels([f"«{x}»" for x in w], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.3)
    ax.set_xlabel("peso IDF", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Palabras de las 12 fichas", 9.6)
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c47_idf.png")


def f_tfidf_v11():
    V = NUM["v11"]
    fig, ax = plt.subplots(figsize=(W_FIG, 2.7))
    ws = [f["palabra"] for f in V["comunes"]]
    x = np.arange(len(ws))
    wd = 0.27
    ax.bar(x - wd, [f["norm_q"] for f in V["comunes"]], wd, color=BLUE, label="peso en la consulta V11 (normalizado)")
    ax.bar(x, [f["norm_d"] for f in V["comunes"]], wd, color=AMBER, label="peso en la ficha D11 (normalizado)")
    ax.bar(x + wd, [f["prod"] for f in V["comunes"]], wd, color=C47, label="producto (aporte al coseno)")
    for k, f in enumerate(V["comunes"]):
        ax.text(k + wd, f["prod"] + 0.012, f"{f['prod']:.3f}", ha="center", fontsize=8.6, color=C47, fontweight="bold")
        ax.text(k - wd, f["norm_q"] + 0.012, f"{f['norm_q']:.3f}", ha="center", fontsize=7.8, color=BLUE)
        ax.text(k, f["norm_d"] + 0.012, f"{f['norm_d']:.3f}", ha="center", fontsize=7.8, color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels([f"«{w}»\nDF = {f['df']}" for w, f in zip(ws, V["comunes"])], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("peso", fontsize=9)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")
    titulo(ax, f"V11 frente a D11 con TF-IDF: solo {len(ws)} palabras comunes; la suma de los productos es {V['suma']:.3f}", 9.6)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c47_tfidf_v11.png")


def f_coseno2d():
    g = NUM["geom"]
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 3.5))
    ax = axs[0]
    for nombre, v, col in (("q", g["q"], INK), ("A", g["a"], BLUE), ("B", g["b"], AMBER)):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4 if nombre == "B" else 2), zorder=2 if nombre == "B" else 3)
        if nombre == "q":
            ax.text(0.0, 1.75, "q = (1, 1)", fontsize=8.8, color=col, fontweight="bold")
            ax.plot([0.4, 0.95], [1.65, 1.05], color=GRAY, lw=0.8)
        else:
            ax.text(v[0] + 0.1, v[1] + 0.12, f"{nombre} = ({v[0]:g}, {v[1]:g})", fontsize=8.8, color=col, fontweight="bold")
    ax.plot([1, 2], [1, 0], color=BLUE, ls=":", lw=1.2)
    ax.plot([1, 4], [1, 4], color=AMBER, ls=":", lw=1.2, alpha=0.0)
    ax.text(1.65, 0.62, f"{g['dist_a']:.3f}", color=BLUE, fontsize=8.4)
    ax.text(2.55, 2.0, f"{g['dist_b']:.3f}", color=AMBER, fontsize=8.4)
    ax.plot([1, 4], [1, 4], color=AMBER, ls=":", lw=1.0)
    ax.set_xlim(-0.3, 5.6)
    ax.set_ylim(-0.3, 4.8)
    ax.set_aspect("equal")
    ax.set_xlabel("veces «clave»", fontsize=9)
    ax.set_ylabel("veces «recuperar»", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Conteos: B es lejano pero apunta igual que q", 9.2)
    style(ax)
    ax = axs[1]
    th = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5)
    for nombre, v, col in (("q y B", g["unit_q"], INK), ("A", g["unit_a"], BLUE)):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
    ax.text(0.75, 0.82, "q y B\n(0.707, 0.707)", fontsize=8.8, color=INK, fontweight="bold", ha="left")
    ax.text(0.7, -0.09, "A (1, 0)", fontsize=8.8, color=BLUE, fontweight="bold", ha="center")
    ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=0, theta2=45, color=RED, lw=1.8))
    ax.text(0.42, 0.13, f"45°\ncos = {g['cos_a']:.4f}", fontsize=8.4, color=RED)
    ax.text(0.2, 0.62, "0° entre q y B\ncos = 1", fontsize=8.4, color=INK, ha="center")
    ax.set_xlim(-0.12, 1.3)
    ax.set_ylim(-0.18, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("clave (normalizada)", fontsize=9)
    ax.set_ylabel("recuperar (normalizada)", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Normalizados: lo que queda es la dirección", 9.2)
    style(ax)
    fig.tight_layout()
    save(fig, "c47_geometria.png")


def f_producto384():
    v = NUM["v01"]
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    ax = axs[0]
    c = calc()
    prod = c["Q"][c["cid"].index("V01")] * c["D"][0]
    ax.bar(np.arange(1, 17), prod[:16], color=[C47 if x > 0 else RED for x in prod[:16]])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("coordenada (primeras 16 de 384)", fontsize=9)
    ax.set_ylabel("producto qⱼ · dⱼ", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Cada coordenada aporta un poquito", 9.4)
    style(ax, "y")
    ax = axs[1]
    ax.plot(np.arange(1, 385), v["acum"], color=C47, lw=2)
    ax.axhline(v["suma"], color=AMBER, ls="--", lw=1.3)
    ax.text(12, v["suma"] - 0.05, f"suma final = {v['suma']:.4f}", color=AMBER, fontsize=8.8, fontweight="bold")
    ax.set_xlabel("coordenadas sumadas (de 1 a 384)", fontsize=9)
    ax.set_ylabel("suma acumulada", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "La suma de los 384 productos es el coseno", 9.4)
    style(ax)
    fig.tight_layout()
    save(fig, "c47_producto384.png")


def f_ranking_v01():
    c = calc()
    v = NUM["v01"]
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 3.7), sharex=True)
    ids = c["ids"]
    for ax, (nombre, sc) in zip(axs, (("TF-IDF (cuenta palabras)", v["tfidf"]), ("MiniLM (embedding)", v["minilm"]))):
        orden = sorted(range(12), key=lambda j: (-sc[j], ids[j]))
        cols = [GREEN if ids[j] == "D01" else GRAY for j in orden]
        ax.barh(range(12), [sc[j] for j in orden], color=cols)
        for k, j in enumerate(orden):
            ax.text(max(sc[j], 0) + 0.008, k, f"{sc[j]:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(12))
        ax.set_yticklabels([ids[j] for j in orden], fontsize=8.8)
        ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-0.2, 0.58)
        ax.set_xlabel("coseno con V01", fontsize=9)
        ax.tick_params(labelsize=8.6)
        titulo(ax, nombre, 9.6)
        style(ax, "x")
    fig.suptitle("V01 «Olvidé mi clave…»: la ficha correcta (D01, verde) queda 6.ª con TF-IDF y 1.ª con MiniLM", fontsize=9.6, color=C47, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "c47_ranking_v01.png")


def f_mapa(clave, nombre, archivo, cmap, vmin, vmax):
    c = calc()
    S = np.array(NUM["puntajes"][clave])
    cid, ids = c["cid"], c["ids"]
    fig, ax = plt.subplots(figsize=(W_FIG, 5.2))
    im = ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            val = S[i, j]
            dark = (val - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, f"{val:.2f}".replace("-0.00", "0.00") if abs(val) >= 0.005 else "0", ha="center", va="center", fontsize=6.6, color="white" if dark else INK)
        rel = c["cons"][i]["relevante"]
        if rel:
            j = ids.index(rel)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GREEN, lw=2.2))
    ax.set_xticks(range(12))
    ax.set_xticklabels(ids, fontsize=8.4)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f"{q} · {'sin respuesta' if not c['cons'][k]['relevante'] else c['cons'][k]['relevante']}" for k, q in enumerate(cid)], fontsize=8.2)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(nombre, loc="left", fontsize=9.8, color=C47, fontweight="bold", pad=24)
    fig.tight_layout()
    save(fig, archivo)


def f_rangos():
    R = NUM["rangos"]
    cons = R["cons"]
    x = np.arange(len(cons))
    fig, axs = plt.subplots(2, 1, figsize=(W_FIG, 4.9), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax = axs[0]
    wd = 0.38
    rt, rm = R["tfidf"], R["minilm"]
    ax.bar(x - wd / 2, rt, wd, color=AMBER, label="TF-IDF")
    ax.bar(x + wd / 2, rm, wd, color=C47, label="MiniLM")
    for k in range(len(cons)):
        ax.text(k - wd / 2, rt[k] + 0.2, str(rt[k]), ha="center", fontsize=8.4, color=AMBER, fontweight="bold")
        ax.text(k + wd / 2, rm[k] + 0.2, str(rm[k]), ha="center", fontsize=8.4, color=C47, fontweight="bold")
    ax.axhline(3.5, color=RED, ls="--", lw=1)
    ax.text(len(cons) - 0.5, 3.75, "k = 3: lo que cabe en la pantalla", color=RED, fontsize=8.4, ha="right")
    ax.set_ylim(0, 12.8)
    ax.set_ylabel("rango r de la ficha correcta\n(1 = primera)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper left")
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Posición de la ficha correcta en cada una de las 12 consultas respondibles (menos es mejor)", 9.4)
    style(ax, "y")
    ax = axs[1]
    ax.bar(x - wd / 2, [1 / r for r in rt], wd, color=AMBER)
    ax.bar(x + wd / 2, [1 / r for r in rm], wd, color=C47)
    ax.axhline(NUM["hit"]["mrr_tfidf"], color=AMBER, ls="--", lw=1.6, label=f"MRR TF-IDF = {NUM['hit']['mrr_tfidf']:.3f}")
    ax.axhline(NUM["hit"]["mrr_minilm"], color=C47, ls="--", lw=1.6, label=f"MRR MiniLM = {NUM['hit']['mrr_minilm']:.3f}")
    ax.set_ylim(0, 1.42)
    ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper center")
    ax.set_ylabel("rango recíproco 1/r", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n{d}" for q, d in zip(cons, R["relevante"])], fontsize=8.2)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "El MRR es el promedio de las barras de abajo (línea punteada)", 9.4)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c47_rangos.png")


def f_hitk():
    H = NUM["hit"]
    k = np.array(H["k"])
    n = H["n"]
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.plot(k, np.array(H["tfidf"]) / n, "o-", color=AMBER, lw=2, label="TF-IDF")
    ax.plot(k, np.array(H["minilm"]) / n, "o-", color=C47, lw=2, label="MiniLM")
    for kk in (1, 3):
        ax.annotate(f"{H['minilm'][kk - 1]}/{n}", (kk, H["minilm"][kk - 1] / n), (kk + 0.1, H["minilm"][kk - 1] / n - 0.11), fontsize=8.4, color=C47, fontweight="bold")
        ax.annotate(f"{H['tfidf'][kk - 1]}/{n}", (kk, H["tfidf"][kk - 1] / n), (kk + 0.15, H["tfidf"][kk - 1] / n - 0.1), fontsize=8.4, color=AMBER, fontweight="bold")
    ax.axvline(3, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("k: cuántas fichas mostramos", fontsize=9)
    ax.set_ylabel(f"Hit@k  (de {n} respondibles)", fontsize=9)
    ax.set_xticks(k)
    ax.set_ylim(0.25, 1.05)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")
    titulo(ax, "Mostrar más fichas siempre ayuda... hasta cierto punto", 9.4)
    style(ax)
    ax = axs[1]
    gt = np.diff([0] + H["tfidf"])
    gm = np.diff([0] + H["minilm"])
    ax.bar(k - 0.2, gt, 0.4, color=AMBER, label="TF-IDF")
    ax.bar(k + 0.2, gm, 0.4, color=C47, label="MiniLM")
    ax.set_xlabel("k", fontsize=9)
    ax.set_ylabel("consultas que se ganan al pasar de k − 1 a k", fontsize=8.4)
    ax.set_xticks(k)
    ax.tick_params(labelsize=8.4)
    ax.legend(frameon=False, fontsize=8.6)
    titulo(ax, "Lo que gana cada ficha extra", 9.4)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c47_hitk.png")


def f_pca():
    P = NUM["pca"]
    c = calc()
    fig, ax = plt.subplots(figsize=(W_FIG, 5.0))
    for q, xy in P["consultas"].items():
        k = c["cid"].index(q)
        rel = c["cons"][k]["relevante"]
        if rel:
            tx = P["fichas"][rel]
            ax.plot([xy[0], tx[0]], [xy[1], tx[1]], color=GREEN, lw=0.7, alpha=0.45, zorder=1)
    for i, xy in P["fichas"].items():
        ax.scatter(*xy, s=70, color=C47, zorder=3)
        off = {"D11": (-0.085, -0.05), "D12": (0.012, 0.03), "D02": (0.015, -0.055), "D05": (-0.075, -0.06), "D01": (0.015, -0.055)}.get(i, (0.012, 0.02))
        ax.annotate(i, xy, (xy[0] + off[0], xy[1] + off[1]), fontsize=9, color=C47, fontweight="bold")
    for q, xy in P["consultas"].items():
        sin = not c["cons"][c["cid"].index(q)]["relevante"]
        ax.scatter(*xy, s=26, marker="x" if not sin else "s", color=AMBER if not sin else RED, zorder=3, lw=1.3)
        off = {"V05": (-0.075, 0.02), "V11": (0.012, 0.02), "V12": (0.012, -0.05), "V01": (0.012, -0.05)}.get(q, (0.01, -0.045))
        ax.annotate(q, xy, (xy[0] + off[0], xy[1] + off[1]), fontsize=7.2, color=AMBER if not sin else RED)
    ax.scatter([], [], s=70, color=C47, label="fichas D01-D12 (con ellas se calcula el PCA)")
    ax.scatter([], [], s=26, marker="x", color=AMBER, label="consultas con respuesta (proyectadas)")
    ax.scatter([], [], s=26, marker="s", color=RED, label="consultas sin respuesta (proyectadas)")
    ax.plot([], [], color=GREEN, lw=0.8, label="consulta → su ficha correcta")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel(f"PC1 ({P['prop'][0] * 100:.1f} % de la variación)", fontsize=9)
    ax.set_ylabel(f"PC2 ({P['prop'][1] * 100:.1f} % de la variación)", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, f"Las 384 coordenadas resumidas en 2 ejes: solo {P['suma'] * 100:.1f} % de la variación", 9.6)
    style(ax)
    fig.tight_layout()
    save(fig, "c47_pca.png")


def f_codigos():
    C = NUM["codigos"]
    c = calc()
    fig = plt.figure(figsize=(W_FIG, 3.9))
    ax = fig.add_axes([0, 0.52, 1, 0.48])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.text(1, 18.6, "D11 y D12: casi la misma frase; cambia solo el código del proyecto", fontsize=10, fontweight="bold", color=C47, va="center")
    for k, (d, y) in enumerate((("D11", 12.2), ("D12", 5.0))):
        txt = next(x["texto"] for x in c["docs"] if x["id"] == d)
        cod = "A17" if d == "D11" else "A71"
        fecha = "12 de noviembre" if d == "D11" else "19 de noviembre"
        ax.add_patch(FancyBboxPatch((1, y - 2.6), 98, 5.4, boxstyle="round,pad=0.2,rounding_size=1.0", fc="#F8FAFC", ec=GRAY, lw=1.1))
        ax.text(3, y + 0.3, d, fontsize=9.6, fontweight="bold", color=C47, va="center")
        ax.text(10, y + 0.3, "El proyecto", fontsize=9, va="center")
        ax.text(23.5, y + 0.3, cod, fontsize=10, fontweight="bold", va="center", color=RED, bbox=dict(fc="#FEE2E2", ec=RED, boxstyle="round,pad=0.25"))
        ax.text(29.5, y + 0.3, "se entrega el", fontsize=9, va="center")
        ax.text(43, y + 0.3, fecha, fontsize=9.4, fontweight="bold", va="center", color=RED)
        ax.text(60.5, y + 0.3, "de 2026 a las 18:00. Incluye un informe PDF…", fontsize=9, va="center")
    ax = fig.add_axes([0.08, 0.1, 0.9, 0.34])
    labels = ["V11 con TF-IDF", "V11 con MiniLM", "V12 con TF-IDF", "V12 con MiniLM"]
    pairs = [(C["V11"]["tfidf_rel"], C["V11"]["tfidf_otro"]), (C["V11"]["minilm_rel"], C["V11"]["minilm_otro"]),
             (C["V12"]["tfidf_rel"], C["V12"]["tfidf_otro"]), (C["V12"]["minilm_rel"], C["V12"]["minilm_otro"])]
    y = np.arange(4)
    ax.barh(y - 0.19, [p[0] for p in pairs], 0.36, color=GREEN, label="ficha correcta")
    ax.barh(y + 0.19, [p[1] for p in pairs], 0.36, color=RED, label="la otra (A71/A17)")
    for k, p in enumerate(pairs):
        ax.text(p[0] + 0.01, k - 0.19, f"{p[0]:.3f}", va="center", fontsize=8.4, color=GREEN)
        ax.text(p[1] + 0.01, k + 0.19, f"{p[1]:.3f}", va="center", fontsize=8.4, color=RED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("coseno", fontsize=9)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")
    style(ax, "x")
    save(fig, "c47_codigos.png")


def f_umbral():
    M = NUM["maximos"]
    c = calc()
    fig, axs = plt.subplots(1, 2, figsize=(W_FIG, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    mx = np.array(M["minilm"])
    con = np.array(M["con_respuesta"])
    order = np.argsort(mx)
    for k, i in enumerate(order):
        ax.barh(k, mx[i], color=GREEN if con[i] else RED)
    for ct in (0.3, 0.5):
        ax.axvline(ct, color=INK, ls="--", lw=1.1)
        ax.text(ct + 0.005, 16.1, f"corte {ct}", fontsize=8.2, color=INK)
    ax.set_yticks(range(16))
    ax.set_yticklabels([M["cons"][i] for i in order], fontsize=8)
    ax.set_xlim(0, 0.86)
    ax.set_ylim(-0.6, 17)
    ax.set_xlabel("coseno máximo de MiniLM de la consulta", fontsize=9)
    ax.scatter([], [], marker="s", color=GREEN, label="con respuesta (12)")
    ax.scatter([], [], marker="s", color=RED, label="sin respuesta (4)")
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Un corte alto también deja fuera consultas con respuesta", 9.0)
    style(ax, "x")
    ax = axs[1]
    U = NUM["umbral"]
    xs = np.arange(4)
    ax.bar(xs - 0.2, [u["resp_ac"] for u in U], 0.4, color=GREEN, label="con respuesta aceptadas (de 12)")
    ax.bar(xs + 0.2, [u["sin_ac"] for u in U], 0.4, color=RED, label="sin respuesta aceptadas (de 4)")
    for k, u in enumerate(U):
        ax.text(k - 0.2, u["resp_ac"] + 0.25, str(u["resp_ac"]), ha="center", fontsize=8.8, color=GREEN, fontweight="bold")
        ax.text(k + 0.2, u["sin_ac"] + 0.25, str(u["sin_ac"]), ha="center", fontsize=8.8, color=RED, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{u['corte']}" for u in U], fontsize=9)
    ax.set_xlabel("corte sobre el coseno máximo", fontsize=9)
    ax.set_ylabel("consultas aceptadas", fontsize=9)
    ax.set_ylim(0, 14.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Costo de cada corte", 9.4)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c47_umbral.png")


PASOS = dict(etapas=f_pipeline, conteo=f_conteo, sin_palabras=f_sin_palabras, idf=f_idf, tfidf_v11=f_tfidf_v11, geometria=f_coseno2d, producto384=f_producto384,
             ranking_v01=f_ranking_v01, mapa_tfidf=lambda: f_mapa("tfidf", "Coseno TF-IDF · 16 consultas × 12 fichas (verde = ficha correcta)", "c47_mapa_tfidf.png", "Blues", 0, 0.5),
             mapa_minilm=lambda: f_mapa("minilm", "Coseno MiniLM · 16 consultas × 12 fichas (verde = ficha correcta)", "c47_mapa_minilm.png", "BuPu", -0.2, 0.7),
             rangos=f_rangos, hitk=f_hitk, pca=f_pca, codigos=f_codigos, umbral=f_umbral)

if __name__ == "__main__":
    numeros()
    for k, fn in PASOS.items():
        fn()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c47_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("todas las cifras de las diapositivas reproducidas:", NUM["todo_igual"])
    for r in NUM["comparacion"]:
        print(("OK  " if r["igual"] else "DIF ") + r["item"], "|", r["diapositiva"], "|", r["reproducido"])
    print("ok")
