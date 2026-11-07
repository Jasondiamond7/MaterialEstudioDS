"""Clase 35 · serie completa de videos "profesor experto" (Random Forest y Bagging) + video completo.

Usa el motor video_profe.py (con las opciones opt-in NUM_DECIMAL y SIMBOLOS_VOZ) y los guiones guion_c35_NN_*.json.
Ejecutar con el venv de Guias/videos/.venv:
    python video_c35_completo.py                       # construye los que falten + el video completo
    python video_c35_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c35_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c35_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c35_completo.py --base 2              # audio "antes": motor por defecto + pronunciaciones de C34
                                                       # (en _tmp_c35_profe/base_NN, para medir la mejora con Whisper)
    python video_c35_completo.py --completo            # solo concatena el video completo
"""
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import video_profe as vp  # noqa: E402

# Pronunciaciones de la Clase 34 (linea base "antes"), copiadas tal cual de video_c34_completo.py.
PRON_C34 = [
    (r"min_samples_leaf", "min sámpols lif"), (r"min_samples_split", "min sámpols split"),
    (r"max_leaf_nodes", "max lif nóuds"), (r"max_depth", "max dépz"), (r"max_features", "max fíchurs"),
    (r"ccp_alpha", "ce ce pe alfa"), (r"feature_importances_", "fíchur impórtanses"),
    (r"permutation_importance", "permiutéishon impórtans"), (r"export_text", "export text"),
    (r"DecisionTreeClassifier", "disíshon tri clásifaier"), (r"RandomForestClassifier", "rándom fórest clásifaier"),
    (r"GridSearchCV", "grid serch ce ve"), (r"cross_val_score", "cros val escór"),
    (r"_", " "),
    (r"frec", "frecuencia de"), (r"n contactos", "número de contactos"), (r"numero fallas", "número de fallas"),
    (r"clv", "ce ele ve"), (r"CLV", "ce ele ve"), (r"Gini", "yíni"), (r"gini", "yíni"),
    (r"entropy", "éntropi"), (r"AUC", "a u ce"), (r"CART", "cart"), (r"lift", "lift"),
    (r"Random Forest", "rándom fórest"), (r"Gradient Boosting", "gréidient bústing"),
    (r"boosting", "bústing"), (r"bootstrap", "bútstrap"), (r"Bootstrap", "Bútstrap"),
    (r"out-of-bag", "aut of bag"), (r"sklearn", "saikit lern"), (r"random state", "rándom estéit"),
    (r"churn", "chern"), (r"telco", "télco"), (r"target", "tárget"), (r"dataset", "dáta set"),
    (r"folds?", "fóld"), (r"train", "tréin"), (r"test", "test"), (r"score", "escór"),
    (r"embedded", "embéded"), (r"filter", "fílter"), (r"wrapper", "uráper"), (r"marketing", "márketing"),
    (r"call center", "col cénter"), (r"sqrt", "raíz cuadrada"), (r"log1p", "log uno pe"),
    (r"LLM", "ele ele eme"), (r"PCA", "pe ce a"), (r"ML", "eme ele"), (r"rpart", "erre part"),
    (r"α", "alfa"), (r"ρ", "ro"), (r"σ", "sigma"), (r"shuffle", "shófol"), (r"scorecard", "escór card"),
]

# Pronunciaciones de la Clase 35. Orden: nombres compuestos primero, luego la regla general "_" -> " ", luego palabras.
# Se aplican DESPUES de video_comun.PRONUNCIACION (que ya convirtio RMSE, one-hot, scikit-learn...) y de los numeros.
# Cada respelling marcado (W) se eligio con la vuelta de Whisper: _tmp_c35_profe/lab (frases portadoras, A/B de candidatos)
# y _tmp_c35_profe/whisper (secciones completas). Las comas fuerzan una micro-pausa que evita que la voz funda las letras.
PRON_C35 = [
    # nombres de la API (antes de la regla "_")
    (r"n_estimators", "en estímeitors"), (r"max_features", "max fíchers"),                      # (W) fíchurs -> «fetches»
    (r"max_samples", "max sámpols"), (r"min_samples_leaf", "min sámpol lif"),                    # (W) sámpols lif -> «sample sleeve»
    (r"min_samples_split", "min sámpol split"), (r"max_depth", "max dépth"),
    (r"class_weight", "clas ueit"),                                                             # (W) clas uéit -> «class uait»
    (r"oob_score_?", ", o, o, bé, escór,"),                                                     # (W) o o be escór -> «oobi escorre»
    (r"n_jobs", "en yobs"), (r"feature_importances_", "fícher impórtanses"),
    (r"permutation_importance", "permiutéishon impórtans"), (r"estimators_samples_", "estímeitors sámpols"),
    (r"random_state", "rándom estéit"),
    (r"BaggingClassifier", "báguin clásifaier"), (r"BaggingRegressor", "báguin rigréser"),
    (r"RandomForestClassifier", "rándom fórest clásifaier"), (r"RandomForestRegressor", "rándom fórest rigréser"),
    (r"DecisionTreeClassifier", "disíshon tríi clásifaier"), (r"DecisionTreeRegressor", "disíshon tríi rigréser"),  # (W)
    (r"RandomizedSearchCV", "rándomaisd sérch, ce uve"), (r"GridSearchCV", "gríd sérch, ce uve"),                    # (W)
    (r"Gr_Liv_Area", "área habitable"), (r"Sale_Price", "precio de venta"),
    (r"_", " "),
    # siglas y simbolos de codigo
    (r"\s*\bOOB\b", ", o, o, bé,"),                                                             # (W) o o be -> «lob», «oobi»
    (r"\s*\bAUC\b", ", a, u, ce,"),                                                             # (W) un a u ce -> «una UC»
    (r"erre eme ese e", "erre, eme, ese, e"),                                                   # (W) -> «RMS»
    (r"\bMDI\b", "eme de i"), (r"\bROC\b", "roc"), (r"\bmtry\b", "em, trai,"),  # (W) em trai -> «entra y»
    (r"'sqrt'", "raíz cuadrada"), (r"\bsqrt\b", "raíz cuadrada"), (r"'log2'", "logaritmo en base dos"),
    (r"CIFAR-diez", "sífar diez"), (r"\bCV\b", "ce ve"), (r"\bRGB\b", "erre ge be"), (r"\bRF\b", "erre efe"),
    # terminos en ingles
    (r"\s*\bout-of-bag\b", ", áut óf bág"), (r"\bOut-of-bag\b", "Áut óf bág"),                 # (W) áut of bag -> «UTF bag»
    (r"\bRandom Forest\b", "rándom fórest"), (r"\bRandom Forests\b", "rándom fórests"),
    (r"\bbagging\b", "báguin"), (r"\bBagging\b", "Báguin"),
    (r"\bbootstrap aggregating\b", "bútstrap ágregueitin"), (r"\baggregating\b", "ágregueitin"),
    (r"\bbootstrap\b", "bútstrap"), (r"\bBootstrap\b", "Bútstrap"),
    (r"\bGradient Boosting\b", "gréidient bústing"), (r"\bboosting\b", "bústing"), (r"\bBoosting\b", "Bústing"),
    (r"\brandom subspace\b", "rándom, sábspeis"), (r"\bhold-out\b", "jóld áut"),                # (W)
    (r"\bfeatures\b", "fíchers"), (r"\bfeature\b", "fícher"), (r"\bdataset\b", "dáta set"),
    (r"\btrain\b", "tréin"), (r"\bscore\b", "escór"), (r"\bthreshold\b", "zréshold"),
    (r"\bHeart\b", "Jart"), (r"\bAmes\b", "Eims"), (r"\bHousing\b", "Jáusin"),                 # (W) Éims -> «EINDS»
    (r"\bThal\b", "Tal"), (r"\bChestPain\b", "chést péyn"),                                    # (W) chest péin -> «Just Paint»
    (r"\bBreiman\b", "Bráiman"), (r"\bEfron\b", "Éfron"),
    (r"\bsklearn\b", "sáikit lern"), (r"saikit lern", "sáikit lern"),                           # (W) -> «Psykit Learn»
    (r"uan jot", "uán jót"),                                                                    # (W) -> «one hop», «and y»
    (r"\bmarketing\b", "márketing"),
    (r"ρ", "ro"), (r"σ", "sigma"),
]

vp.NUM_DECIMAL = True
vp.SIMBOLOS_VOZ = True
vp.PRONUNCIACION_EXTRA.extend(PRON_C35)

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C35"
TMP = VIDEOS_DIR / "_tmp_c35_profe"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C35_01_ArbolInestable", "guion_c35_01_inestable.json", "Introducción: un árbol inestable"),
    (2, "C35_02_Bootstrap", "guion_c35_02_bootstrap.json", "Bootstrap a mano"),
    (3, "C35_03_OOB", "guion_c35_03_oob.json", "Fuera de bolsa: el error OOB"),
    (4, "C35_04_Bagging", "guion_c35_04_bagging.json", "Bagging: votar y promediar"),
    (5, "C35_05_Varianza", "guion_c35_05_varianza.json", "Por qué promediar reduce la varianza"),
    (6, "C35_06_RandomForest", "guion_c35_06_randomforest.json", "Random Forest: azar en las variables"),
    (7, "C35_07_MaxFeatures", "guion_c35_07_maxfeatures.json", "max_features en la práctica"),
    (8, "C35_08_Hiperparametros", "guion_c35_08_hiperparametros.json", "Número de árboles e hiperparámetros"),
    (9, "C35_09_ImportanciaLimites", "guion_c35_09_importancia.json", "Importancia y límites"),
    (10, "C35_10_Comparacion", "guion_c35_10_comparacion.json", "La comparación final"),
    (11, "C35_11_Repaso", "guion_c35_11_repaso.json", "Repaso final y checklist"),
]
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def modo_base():
    """Linea base para medir la pronunciacion: motor por defecto + solo las pronunciaciones de C34."""
    vp.NUM_DECIMAL = False
    vp.SIMBOLOS_VOZ = False
    for par in PRON_C35:
        vp.PRONUNCIACION_EXTRA.remove(par)
    vp.PRONUNCIACION_EXTRA.extend(PRON_C34)


def construir_uno(n, solo_audio=False, tmp=None):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4" if tmp is None else Path(tmp) / f"{nombre}.mp4"   # --base no toca C35/
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = tmp or TMP / f"{n:02}"
    return vp.construir(AQUI / guion, salida, tmp, solo_audio=solo_audio)


def _dur(ruta, sel="v:0"):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "stream=duration",
                          "-of", "csv=p=0", str(ruta)], capture_output=True, text=True, check=True).stdout.strip()
    return float(out.split()[0])


def _secciones(n, total):
    """[(nombre, inicio)] de las secciones internas del video n (para sub-dividir capitulos largos)."""
    inf = (TMP / f"{n:02}") / "informe.json"
    if not inf.exists():
        return [("", 0.0)]
    secs = json.loads(inf.read_text(encoding="utf-8"))["secciones"]
    return [(s[0], 0.0 if i == 0 else float(s[1])) for i, s in enumerate(secs)] or [("", 0.0)]


def _srt_t(s):
    h, m, r = s.split(":")
    sec, ms = r.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def _srt_f(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def completo():
    TMP.mkdir(parents=True, exist_ok=True)
    work = TMP / "completo"
    work.mkdir(exist_ok=True)
    archivos = [SALIDA / f"{s[1]}.mp4" for s in SERIE]
    faltan = [a for a in archivos if not a.exists()]
    if faltan:
        raise SystemExit(f"faltan videos: {[a.name for a in faltan]}")
    durs = [_dur(a) for a in archivos]
    (work / "lista.txt").write_text("\n".join(f"file '{a.as_posix()}'" for a in archivos), encoding="utf-8")
    video = work / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "lista.txt"),
                    "-map", "0:v", "-c", "copy", "-video_track_timescale", "1000", str(video)], check=True)
    wav = work / "audio.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        for a, d in zip(archivos, durs):
            pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(a), "-vn", "-ac", "1", "-ar", "48000",
                                  "-f", "s16le", "-"], capture_output=True, check=True).stdout
            n = int(round(d * 48000)) * 2
            pcm = pcm[:n] + b"\x00" * max(0, n - len(pcm))
            w.writeframes(pcm)
    srt_out, k, t0 = [], 1, 0.0
    meta = [";FFMETADATA1", "title=Clase 35 · Random Forest y Bagging (completo)", "artist=Prof. Bastián Galasso"]
    for (n, nombre, _g, titulo), a, d in zip(SERIE, archivos, durs):
        for bloque in a.with_suffix(".srt").read_text(encoding="utf-8").strip().split("\n\n"):
            ls = bloque.strip().split("\n")
            if len(ls) < 3:
                continue
            ini, fin = ls[1].split(" --> ")
            srt_out.append(f"{k}\n{_srt_f(t0 + _srt_t(ini))} --> {_srt_f(t0 + _srt_t(fin))}\n" + "\n".join(ls[2:]) + "\n")
            k += 1
        secs = _secciones(n, d)
        cand = [s_ for _n, s_ in secs[1:]]
        partes = max(1, math.ceil(d / MAX_CAP))
        cortes = [0.0]
        for kk in range(1, partes):
            objetivo = d * kk / partes
            libres = [c for c in cand if c > cortes[-1] + 30]
            cortes.append(min(libres, key=lambda c: abs(c - objetivo)) if libres else objetivo)
        cortes.append(d)
        for j in range(len(cortes) - 1):
            nom = f"{n}. {titulo}" + (f" (parte {j + 1}/{len(cortes) - 1})" if len(cortes) > 2 else "")
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round((t0 + cortes[j]) * 1000))}",
                     f"END={int(round((t0 + cortes[j + 1]) * 1000))}", f"title={nom}"]
        t0 += d
    srt = SALIDA / "C35_Completo_RandomForest.srt"
    srt.write_text("\n".join(srt_out), encoding="utf-8")
    (work / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    salida = SALIDA / "C35_Completo_RandomForest.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-i", str(srt),
                    "-i", str(work / "meta.txt"), "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3",
                    "-map_chapters", "3", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=spa", "-metadata:s:a:0", "language=spa", "-t", f"{t0:.3f}",
                    "-movflags", "+faststart", str(salida)], check=True)
    print(f"OK {salida}  ({t0 / 60:.2f} min)")


def _nums(args):
    out = []
    for a in args:
        if a.startswith("--"):
            break
        out.append(int(a))
    return out


if __name__ == "__main__":
    av = sys.argv[1:]
    if av and av[0] == "--previa":
        n = int(av[1])
        tiempos = [float(x) for x in av[2:]]
        tmp = TMP / f"{n:02}"
        construir_uno(n, solo_audio=True)
        for f in vp.previsualizar(tmp, tiempos, tmp / "previa"):
            print(f)
    elif av and av[0] == "--audio":
        for n in _nums(av[1:]):
            L, inf = construir_uno(n, solo_audio=True)
            print(n, f"{inf['duracion']:.1f} s", inf["beats"], "beats", [(s, round(t)) for s, t in inf["secciones"]])
    elif av and av[0] == "--base":
        modo_base()
        for n in _nums(av[1:]):
            L, inf = construir_uno(n, solo_audio=True, tmp=TMP / f"base_{n:02}")
            print("base", n, f"{inf['duracion']:.1f} s")
    elif av and av[0] == "--solo":
        for n in _nums(av[1:]):
            construir_uno(n)
    elif av and av[0] == "--completo":
        completo()
    else:
        for s in SERIE:
            if not (SALIDA / f"{s[1]}.mp4").exists():
                construir_uno(s[0])
        completo()
