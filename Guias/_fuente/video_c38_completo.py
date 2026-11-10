"""Clase 38 · serie completa de videos "profesor experto" (Evaluación de modelos supervisados y selección de modelos)
+ video completo.

Usa el motor video_profe.py (con las opciones opt-in NUM_DECIMAL y SIMBOLOS_VOZ) y los guiones guion_c38_NN_*.json.
Ejecutar con el venv de Guias/videos/.venv:
    python video_c38_completo.py                       # construye los que falten + el video completo
    python video_c38_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c38_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c38_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c38_completo.py --base 2              # audio "antes": mismas opciones, pero solo las pronunciaciones de C35-C37
                                                       # (en _tmp_c38_profe/base_NN, para medir la mejora con Whisper)
    python video_c38_completo.py --completo            # solo concatena el video completo
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
import video_c37_completo as R37  # noqa: E402   (reutiliza PRON_C37, que ya incluye C35 y C36; ese import rehace la lista: se rehace abajo)

# Pronunciaciones de la Clase 38. Se aplican DESPUES de video_comun.PRONUNCIACION y de la conversion de numeros.
# Todas con \b explicito (en C37 algunas reglas quedaron con un caracter de retroceso y no actuan: aqui no se heredan
# como reglas nuevas, solo se dejan tal cual estaban). Cada respelling marcado (W) se eligio con la vuelta de Whisper:
# _tmp_c38_profe/lab (frases portadoras reales del guion).
# 1) nombres de la API (antes de las reglas generales)
PRON_C38_API = [
    (r"\bDummyRegressor\b", "Dummy Regressor"), (r"\bDummyClassifier\b", "Dummy Classifier"),      # (W) junto -> «un mi-regresor»
    (r"\bPolynomialFeatures\b", "Polynomial Features"), (r"\bLinearRegression\b", "Linear Regression"),
    (r"\bbrier_score_loss\b", "Brier score loss"),
]
# 2) reglas heredadas que la Clase 38 reemplaza
QUITAR = {r"\bROC\b", r"\s*\bAUC\b"}         # (W) «roc» -> «ROP»; ROC sin respelling -> «ROC»
# 3) palabras de la Clase 38 (despues de las de C35, C36 y C37). Brier, Spambase, UCI, ranger, clone, spam y ROC se dejan
#    tal cual: en el laboratorio (ronda 1) Whisper los escribe bien sin respelling.
PRON_C38_PAL = [
    (r"\bOOF\b", "ou, ou, ef"),                                                    # (W) OOF -> «o OF»; o, o, efe -> «o F»
    (r"\bAUC\b", "a u ce"),                                                        # (W) a, u, ce -> «A, U, C, E»; a u ce -> «AUCE»
    (r"\bAP\b", "a pe"),                                                           # (W) AP -> «app»
    (r"\bridge\b", "Ridch"),                                                     # (W) ridge -> «Wage», «Ganovich»; Ridch -> «Rich»
]
PRON_C38 = PRON_C38_API + [p for p in R37.PRON_C37 if p[0] not in QUITAR] + PRON_C38_PAL

vp.NUM_DECIMAL = True
vp.SIMBOLOS_VOZ = True
vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + PRON_C38

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C38"
TMP = VIDEOS_DIR / "_tmp_c38_profe"
COMPLETO = "C38_Completo_EvaluacionSeleccion"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C38_01_MejorParaQue", "guion_c38_01_mejor.json", "Introducción: ¿mejor modelo para qué?"),
    (2, "C38_02_MetricasRegresion", "guion_c38_02_metricas.json", "MAE, RMSE y R²"),
    (3, "C38_03_ParticionGrupos", "guion_c38_03_particion.json", "Separar por composición: grupos y prueba reservada"),
    (4, "C38_04_AjusteYSeleccion", "guion_c38_04_candidatos.json", "Ajustar no es seleccionar: los candidatos"),
    (5, "C38_05_SesgoVarianza", "guion_c38_05_sesgo.json", "Sobreajuste, sesgo y varianza"),
    (6, "C38_06_ValidacionOOF", "guion_c38_06_oof.json", "Validación cruzada y predicciones fuera de pliegue"),
    (7, "C38_07_RankingCV", "guion_c38_07_ranking.json", "Leer el ranking, la curva y los pliegues"),
    (8, "C38_08_PruebaFinal", "guion_c38_08_prueba.json", "Reajustar, evaluar una vez y diagnosticar"),
    (9, "C38_09_CostoErrores", "guion_c38_09_costo.json", "Clasificación: el costo de cada error"),
    (10, "C38_10_Probabilidades", "guion_c38_10_probabilidades.json", "Umbral, log loss, Brier y calibración"),
    (11, "C38_11_ROCyProtocolo", "guion_c38_11_roc.json", "ROC, precisión y el protocolo del spam"),
    (12, "C38_12_UmbralYRepaso", "guion_c38_12_umbral.json", "Umbral por costo, prueba final y repaso"),
]
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def modo_base():
    """Linea base para medir la pronunciacion: mismas opciones del motor, pero solo las pronunciaciones de C35 a C37."""
    vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + R37.PRON_C37


def construir_uno(n, solo_audio=False, tmp=None):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4" if tmp is None else Path(tmp) / f"{nombre}.mp4"   # --base no toca C38/
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
    meta = [";FFMETADATA1", "title=Clase 38 · Evaluación de modelos supervisados y selección de modelos (completo)",
            "artist=Prof. Eloy Alvarado Narváez"]
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
