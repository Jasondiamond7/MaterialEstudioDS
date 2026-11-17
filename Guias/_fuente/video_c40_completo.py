"""Clase 40 · serie completa de videos "profesor experto" (Naive Bayes y clasificación probabilística) + video completo.

Usa el motor video_profe.py (con las opciones opt-in NUM_DECIMAL y SIMBOLOS_VOZ) y los guiones guion_c40_NN_*.json.
Ejecutar con el venv de Guias/videos/.venv:
    python video_c40_completo.py                       # construye los que falten + el video completo
    python video_c40_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c40_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c40_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c40_completo.py --base 2              # audio "antes": mismas opciones, pero solo las pronunciaciones de C35-C39
                                                       # (en _tmp_c40_profe/base_NN, para medir la mejora con Whisper)
    python video_c40_completo.py --completo            # solo concatena el video completo
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

_DEFECTO = list(vp.PRONUNCIACION_EXTRA)          # pronunciaciones propias del motor (antes de cualquier runner)
import video_c39_completo as R39  # noqa: E402   (reutiliza PRON_C39, que ya incluye C35 a C38; ese import rehace la lista: se rehace abajo)

# Pronunciaciones de la Clase 40. Se aplican DESPUES de video_comun.PRONUNCIACION y de la conversion de numeros.
# Todas con \b explicito. Cada respelling marcado (W) se eligio con la vuelta de Whisper: _tmp_c40_profe/lab
# (frases portadoras reales del guion).
# 1) nombres de la API (antes de las reglas generales)
PRON_C40_API = [
    (r"\bnaivebayes\b", "Naïve Béis"),                                # paquete de R: mismo respelling que Naive Bayes
    (r"\bBernoulliNB\b", "Bernoulli, ene, be"), (r"\bGaussianNB\b", "Gausian, ene, be"),      # (W) GaussianNB -> «GaussianND»
    (r"\bMultinomialNB\b", "multinomial, ene, be"),                   # (W) -> «multinomial N-B»
]
# 2) palabras de la Clase 40 (despues de las de C35 a C39).
PRON_C40_PAL = [
    (r"\bNaive Bayes\b", "Naïve Béis"),     # (W) Naive Bayes -> «Naipas», «AIVBs»; Naïve Béis -> «Naive Bayes», «Naive Base»
    (r"\bBernoulli\b", "Bernouli"),         # (W) Bernoulli -> «ver no huye», «ver Knowji» en frases reales; Bernouli -> «Bernoulli» (6/6)
    (r"\bBayes\b", "Béis"),                # (W) Bayes -> «VAS»; Béis -> «Bayes» (2/2)
    (r"\bLaplace\b", "Lápplas"),            # (W) Laplace -> «la plaza» (6/6); Lápplas -> «Laplace», «laplas»
    (r"\bfree\b", "fríi"),                  # (W) free -> «fría», «FRI»; fríi -> «free» (3/3)
    (r"\baddress\b", "ádres"),              # (W) address -> «a Alvaz»; ádres -> «adres»
    (r"\bSpambase\b", "Spám béis"),         # (W) Spambase -> «Spamases»; Spám béis -> «spambase» (3/3)
    (r"\blog1p\b", "log uno pe"),           # (W) -> «log1p»
]
PRON_C40 = PRON_C40_API + list(R39.PRON_C39) + PRON_C40_PAL

vp.NUM_DECIMAL = True
vp.SIMBOLOS_VOZ = True
vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + PRON_C40

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C40"
TMP = VIDEOS_DIR / "_tmp_c40_profe"
COMPLETO = "C40_Completo_NaiveBayes"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C40_01_TresPreguntas", "guion_c40_01_preguntas.json", "Introducción: previa, condicional y posterior"),
    (2, "C40_02_TeoremaDeBayes", "guion_c40_02_bayes.json", "El teorema de Bayes y la frecuencia de spam"),
    (3, "C40_03_DosSenales", "guion_c40_03_senales.json", "Dos señales, Bernoulli y la explosión de combinaciones"),
    (4, "C40_04_IndependenciaCondicional", "guion_c40_04_independencia.json", "El supuesto naive: independencia dentro de cada clase"),
    (5, "C40_05_SuavizadoLaplace", "guion_c40_05_laplace.json", "Ceros y suavizado de Laplace"),
    (6, "C40_06_FormulaGeneral", "guion_c40_06_formula.json", "Aprender, predecir, K clases y log-probabilidades"),
    (7, "C40_07_Spambase", "guion_c40_07_spambase.json", "Spambase: 54 señales y un correo real"),
    (8, "C40_08_Variantes", "guion_c40_08_variantes.json", "Variantes: Bernoulli, gaussiana y multinomial"),
    (9, "C40_09_ComparacionOOF", "guion_c40_09_oof.json", "Comparar recetas con predicciones fuera de pliegue"),
    (10, "C40_10_DependenciaCalibracion", "guion_c40_10_calibracion.json", "Señales copiadas, calibración y un correo explicado"),
    (11, "C40_11_UmbralCosto", "guion_c40_11_umbral.json", "Decidir: umbral por costo y comprobación histórica"),
    (12, "C40_12_LimitesYRepaso", "guion_c40_12_repaso.json", "Límites, bibliotecas y repaso"),
]
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def modo_base():
    """Linea base para medir la pronunciacion: mismas opciones del motor, pero solo las pronunciaciones de C35 a C39."""
    vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + R39.PRON_C39


def construir_uno(n, solo_audio=False, tmp=None):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4" if tmp is None else Path(tmp) / f"{nombre}.mp4"   # --base no toca C40/
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
    meta = [";FFMETADATA1", "title=Clase 40 · Naive Bayes y clasificación probabilística (completo)",
            "artist=Prof. Eloy Alvarado"]
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
    srt = SALIDA / f"{COMPLETO}.srt"
    srt.write_text("\n".join(srt_out), encoding="utf-8")
    (work / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    salida = SALIDA / f"{COMPLETO}.mp4"
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
