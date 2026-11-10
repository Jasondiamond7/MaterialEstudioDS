"""Vuelta de Whisper: transcribe el audio de narracion de una seccion y lo alinea con el texto enviado al TTS.

Ejecutar con TranscripcionPython/.venv (openai-whisper + CUDA):
    python whisper_check.py N [base|c37] [--modelo medium]
Lee  NN/audio.wav (o base_NN/audio.wav) y whisper/voz_<modo>_NN.json (de dump_voz.py).
Escribe whisper/res_<modo>_NN.json con: WER (claves foneticas), palabras no recuperadas y recuperacion por termino.

Comparacion: ambas cadenas se normalizan (minusculas, sin tildes ni signos) y cada palabra se reduce a una clave
fonetica del espanol (b=v, c/k/q, z/s, gu+e/i, h muda, ll/y, dobles), de modo que "bagging" ~ "báguin". Los numeros
que Whisper escribe en cifras se expanden a palabras eligiendo la lectura que aparece en el texto de referencia.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MODELO = sys.argv[sys.argv.index("--modelo") + 1] if "--modelo" in sys.argv else "medium"
args = [a for a in sys.argv[1:] if not a.startswith("--") and a != MODELO]
NS = [int(a) for a in args if a.isdigit()]
MODO = next((a for a in args if not a.isdigit()), "c37")
N = NS[0] if NS else 1
_MODELO_W = None

# ------------------------------------------------------------------ numeros (copia minima del motor, sin PIL)
_UNI = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece",
        "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
        "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
_DEC = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_CEN = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos", 6: "seiscientos", 7: "setecientos",
        8: "ochocientos", 9: "novecientos"}


def ent(n):
    if n < 30:
        return _UNI[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _DEC[d] + ("" if u == 0 else " y " + _UNI[u])
    if n < 1000:
        c, r = divmod(n, 100)
        return "cien" if (c == 1 and r == 0) else _CEN[c] + ("" if r == 0 else " " + ent(r))
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        base = "mil" if m == 1 else re.sub(r"uno$", "un", ent(m)) + " mil"
        return base + ("" if r == 0 else " " + ent(r))
    m, r = divmod(n, 1_000_000)
    base = "un millón" if m == 1 else re.sub(r"uno$", "un", ent(m)) + " millones"
    return base + ("" if r == 0 else " " + ent(r))


def lecturas(tok):
    """Todas las lecturas plausibles de una cifra escrita por Whisper."""
    out = []
    t = tok
    if re.fullmatch(r"\d+", t):
        out.append(ent(int(t)))
    if re.fullmatch(r"[1-9]\d{0,2}([.,]\d{3})+", t):
        out.append(ent(int(t.replace(".", "").replace(",", ""))))
    m = re.fullmatch(r"(\d+)[.,](\d+)", t)
    if m:
        e, d = m.groups()
        dig = " ".join(_UNI[int(c)] for c in d)
        out.append(f"{ent(int(e))} coma {dig}")
        out.append(f"{ent(int(e))} punto {dig}")
        r = d.lstrip("0")
        if r:
            cola = " ".join(["cero"] * (len(d) - len(r)) + [ent(int(r))])
            out.append(f"{ent(int(e))} coma {cola}")
            out.append(f"{ent(int(e))} punto {cola}")
    return out or [tok]


# ------------------------------------------------------------------ normalizacion y clave fonetica
def norm(w):
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if c.isalnum() and unicodedata.category(c) != "Mn")


def clave(w):
    w = norm(w)
    if not w:
        return w
    w = w.replace("ñ", "ni").replace("oo", "u").replace("ee", "i")
    w = re.sub(r"gu(?=[ei])", "g", w)
    w = re.sub(r"qu(?=[ei])", "k", w)
    w = re.sub(r"c(?=[ei])", "s", w)
    w = w.replace("ch", "X").replace("sh", "X").replace("ll", "y").replace("ph", "f").replace("th", "t")
    w = w.replace("c", "k").replace("q", "k").replace("z", "s").replace("v", "b").replace("w", "u").replace("h", "").replace("j", "")
    w = w.replace("x", "ks").replace("X", "ch")
    w = re.sub(r"ng$", "n", w)
    w = w.replace("y", "i")
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def tokens(t):
    t = re.sub(r"\b([A-Za-z])(\d)\b", r"\1 \2", t)            # P1 -> P 1
    t = t.replace("ρ", " ro ").replace("σ", " sigma ").replace("²", " al cuadrado ").replace("√", " raíz de ")
    t = t.replace("%", " por ciento ").replace("−", " menos ").replace("·", " ")
    return [x for x in re.findall(r"[\w]+(?:[.,]\d+)*", t)]


# ------------------------------------------------------------------ alineacion (Levenshtein por palabras)
def alinear(ref, hyp):
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        bt[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        bt[i][0] = 1
        ri = ref[i - 1]
        for j in range(1, m + 1):
            s = prev[j - 1] + (0 if ri == hyp[j - 1] else 1)
            d = prev[j] + 1
            a = cur[j - 1] + 1
            if s <= d and s <= a:
                cur[j], bt[i][j] = s, 0
            elif d <= a:
                cur[j], bt[i][j] = d, 1
            else:
                cur[j], bt[i][j] = a, 2
        prev = cur
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        b = bt[i][j]
        if i > 0 and j > 0 and b == 0:
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (b == 1 or j == 0):
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return prev[m], ops[::-1]


def canon(words, eq):
    """Reemplaza secuencias (normalizadas) de terminos originales por su respelling: ambos lados usan la misma forma."""
    out, i = [], 0
    while i < len(words):
        for o, r in eq:
            if words[i:i + len(o)] == o:
                out += r
                i += len(o)
                break
        else:
            out.append(words[i])
            i += 1
    return out


def main():
    voz = json.loads((AQUI / "whisper" / f"voz_{MODO}_{N:02}.json").read_text(encoding="utf-8"))
    wav = AQUI / (f"base_{N:02}" if MODO == "base" else f"{N:02}") / "audio.wav"
    vistos, eq = set(), []
    for o, r in voz.get("equivalencias", []):
        k = tuple(norm(x) for x in o.split())
        if k not in vistos:
            vistos.add(k)
            eq.append((k, tuple(norm(x) for x in tokens(r))))
    eq.sort(key=lambda e: -len(e[0]))
    eq = [(list(o), list(r)) for o, r in eq if o != r]
    eq += [(["auc"], ["a", "u", "ce"]), (["oob"], ["o", "o", "be"]), (["rmse"], ["erre", "eme", "ese", "e"]), (["mdi"], ["eme", "de", "i"]),
           (["n"], ["ene"]), (["b"], ["be"]), (["p"], ["pe"]), (["m"], ["eme"]), (["x"], ["equis"]), (["r"], ["erre"]), (["s"], ["ese"]),
           (["l"], ["ele"]), (["f"], ["efe"]), (["k"], ["ka"]), (["d"], ["de"]), (["t"], ["te"]),
           (["cv"], ["ce", "ve"]), (["c"], ["ce"]), (["oobe"], ["o", "o", "be"]), (["oob"], ["o", "o", "be"]), (["oobi"], ["o", "o", "be"]),
           (["auc"], ["a", "u", "ce"]), (["maxfeatures"], ["max", "fíchers"]), (["min", "sample", "leaf"], ["min", "sámpol", "lif"]), (["rho"], ["ro"]), (["randomforest"], ["rándom", "fórest"]), (["emtrai"], ["em", "trai"]),
           (["en", "estimators"], ["en", "estímeitors"]), (["estimators"], ["en", "estímeitors"]), (["en", "jobs"], ["en", "yobs"]),
           (["one", "hot"], ["uán", "jót"]), (["scikit", "learn"], ["sáikit", "lern"]), (["aims"], ["eims"]), (["eams"], ["eims"]), (["auce"], ["a", "u", "ce"]), (["rf"], ["erre", "efe"]), (["f1"], ["efe", "uno"])]
    # C36: variantes de escritura de Whisper para el mismo sonido (homofonos o palabras unidas)
    eq += [(["lite", "gbm"], ["lait", "yi", "bi", "em"]), (["leafwise"], ["leaf", "wise"]), (["levelwise"], ["level", "wise"]),
           (["cev"], ["ce", "uve"]), (["psykit"], ["saikit"]), (["psyche"], ["saikit"]),
           (["gradientboosting"], ["greidien", "boosting"]), (["erly", "stopping"], ["er", "li", "stoping"]),
           (["xgbost"], ["ecs", "yi", "bust"]), (["en", "stimators"], ["en", "estimeitors"])]
    # C36 ronda 2: mismas palabras con otra ortografia (no cambian el sonido); se listan en el informe
    eq += [(["saikit", "learn"], ["saikit", "lern"]), (["search"], ["serch"]), (["veintiuno", "mil"], ["veintiun", "mil"]),
           (["hubert"], ["huber"]), (["exibust"], ["ecs", "yi", "bust"]), (["xgboast"], ["ecs", "yi", "bust"]),
           (["xgburst"], ["ecs", "yi", "bust"]), (["catburst"], ["cat", "boost"])]
    # C37: variantes de escritura de Whisper para el mismo sonido (se listan en el informe)
    eq += [(["w"], ["doble", "u"]), (["e1071"], ["e", "uno", "cero", "siete", "uno"]), (["scaler"], ["skeiler"]), (["glm"], ["ge", "ele", "eme"]),
           (["svc"], ["ese", "uve", "ce"]), (["rbf"], ["erre", "be", "efe"]), (["kernel"], ["kernel"]), (["kernels"], ["kernels"]),
           (["hinch", "loss"], ["jinch", "los"]), (["hinge", "loss"], ["jinch", "los"]),
           (["vn"], ["uve", "ene"]), (["vp"], ["uve", "pe"]), (["nan"], ["nan"]), (["none"], ["nan"]),
           (["predictproba"], ["predict", "proba"]), (["gama"], ["gamma"])]
    # C37 ronda 2: el mismo sonido escrito de otra forma por Whisper (palabras unidas o separadas, letras en vez de su nombre)
    eq += [(["con", "cavidad"], ["concavidad"]), (["iguala"], ["igual", "a"]), (["gridsearch"], ["grid", "serch"]),
           (["ye"], ["y"]), (["z"], ["zeta"]), (["j"], ["jota"]), (["xj"], ["equis", "jota"]), (["v"], ["be"]),
           (["fit", "transform"], ["fit", "transform"])]
    ref_words, ref_frase = [], []
    for k, fr in enumerate(voz["frases"]):
        ws = canon([w for w in (norm(x) for x in tokens(fr["voz"])) if w], eq)
        ref_words += ws
        ref_frase += [k] * len(ws)
    ref_txt = " ".join(ref_words)

    cache = AQUI / "whisper" / f"hyp_{MODO}_{N:02}_{MODELO}.txt"
    if cache.exists() and cache.stat().st_mtime > wav.stat().st_mtime:
        hyp_raw = cache.read_text(encoding="utf-8")
    else:
        global _MODELO_W
        import whisper
        if _MODELO_W is None:
            _MODELO_W = whisper.load_model(MODELO, device="cuda")
        m = _MODELO_W
        r = m.transcribe(str(wav), language="es", fp16=True, temperature=0.0, condition_on_previous_text=False,
                         initial_prompt=None, verbose=None)
        hyp_raw = r["text"]
        cache.write_text(hyp_raw, encoding="utf-8")

    # expandir cifras de la hipotesis con la lectura que aparece en la referencia
    hyp_words = []
    for tk in tokens(hyp_raw):
        if re.search(r"\d", tk):
            cands = [" ".join(norm(x) for x in c.split()) for c in lecturas(tk)]
            elegido = next((c for c in cands if f" {c} " in f" {ref_txt} "), cands[0])
            hyp_words += elegido.split()
        else:
            w = norm(tk)
            if w:
                hyp_words.append(w)
    hyp_words = canon(hyp_words, eq)
    rk, hk = [clave(w) for w in ref_words], [clave(w) for w in hyp_words]
    dist, ops = alinear(rk, hk)
    ok = [False] * len(ref_words)
    malas = {}
    for op, i, j in ops:
        if op == "eq":
            ok[i] = True
        elif op in ("sub", "del"):
            malas.setdefault(ref_words[i], []).append(hyp_words[j] if j is not None else "∅")
    # recuperacion por termino
    terminos = {}
    for k, fr in enumerate(voz["frases"]):
        idx = [i for i, f in enumerate(ref_frase) if f == k]
        toks = [rk[i] for i in idx]
        for term, vt, cnt in fr["terms"]:
            tt = [clave(w) for w in canon([w for w in (norm(x) for x in tokens(vt)) if w], eq)]
            e = terminos.setdefault(term, dict(voz=vt, n=0, recuperado=0, no_ubicado=0, oido=[]))
            e["n"] += cnt
            usados = 0
            for s in range(len(toks) - len(tt) + 1):
                if usados >= cnt or not tt:
                    break
                if toks[s:s + len(tt)] == tt:
                    usados += 1
                    pos = idx[s:s + len(tt)]
                    if all(ok[p] for p in pos):
                        e["recuperado"] += 1
                    else:
                        oido = []
                        for op, i, j in ops:
                            if i in pos and j is not None:
                                oido.append(hyp_words[j])
                        e["oido"].append(" ".join(oido) or "∅")
            e["no_ubicado"] += cnt - usados
    res = dict(seccion=N, modo=MODO, modelo=MODELO, palabras_ref=len(ref_words), palabras_hyp=len(hyp_words),
               errores=dist, wer=round(dist / max(1, len(ref_words)), 4),
               decimales_N_NNN=voz.get("decimales_leidos_como_miles", 0),
               no_recuperadas=sorted(((w, len(v), sorted(set(v))[:4]) for w, v in malas.items()), key=lambda x: -x[1]),
               terminos=terminos)
    (AQUI / "whisper" / f"res_{MODO}_{N:02}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{MODO} {N:02}] WER {res['wer'] * 100:.2f} %  ({dist}/{len(ref_words)})  modelo {MODELO}")
    for term, e in sorted(terminos.items()):
        marca = "OK " if e["recuperado"] == e["n"] - e["no_ubicado"] else "MAL"
        print(f"  {marca} {term:24s} {e['recuperado']}/{e['n'] - e['no_ubicado']}  voz='{e['voz']}'  oido={e['oido'][:3]}")
    print("  palabras no recuperadas (top 25):", [(w, c) for w, c, _ in res["no_recuperadas"][:25]])


if __name__ == "__main__":
    for N in NS:
        main()
