"""Clase 37 · serie completa de videos "profesor experto" (SVM y clasificadores clásicos) + video completo.

Usa el motor video_profe.py (con las opciones opt-in NUM_DECIMAL y SIMBOLOS_VOZ) y los guiones guion_c37_NN_*.json.
Ejecutar con el venv de Guias/videos/.venv:
    python video_c37_completo.py                       # construye los que falten + el video completo
    python video_c37_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c37_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c37_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c37_completo.py --base 2              # audio "antes": mismas opciones, pero solo las pronunciaciones de C35 y C36
                                                       # (en _tmp_c37_profe/base_NN, para medir la mejora con Whisper)
    python video_c37_completo.py --completo            # solo concatena el video completo
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
import video_c36_completo as R36  # noqa: E402   (reutiliza PRON_C36, que ya incluye PRON_C35; ese import rehace la lista: se rehace abajo)

# Pronunciaciones de la Clase 37. Se aplican DESPUES de video_comun.PRONUNCIACION y de la conversion de numeros.
# Cada respelling marcado (W) se eligio con la vuelta de Whisper: _tmp_c37_profe/lab (frases portadoras reales del guion).
# 1) nombres de la API y de R (antes de la regla general "_" -> " " de C35)
PRON_C37_API = [
    (r"predíct_proba", "prédict próba"),                                                     # (W) predíct próba -> «PredicProba»
    (r"LogisticRegression", "Logistic Regression"), (r"SVC", "ese uve ce"),              # (W) ese, uve, ce -> «S, V, se»
    (r"e1071", "e uno cero siete uno"),                                                       # (W) Whisper escribe «E1071»
]
# 2) reglas heredadas que la Clase 37 reemplaza
QUITAR = set()
# 3) palabras de la Clase 37 (despues de las de C35 y C36). SVM, kNN, LDA, QDA, Wisconsin, R, Platt y "hard margin"
#    se dejan tal cual: en el laboratorio (ronda 1) Whisper los escribe bien sin respelling.
PRON_C37_PAL = [
    (r"kernels", "kérnels"), (r"kernel", "kérnel"),                                       # (W) kernel -> «crono o mar»
    (r"RBF", "erre, be, efe"),                                                                # (W) erre be efe -> «rbf» suelto
    (r"hinge loss", "jinch lós"), (r"hinge", "jinch"),                                    # (W) hinge loss -> «hence was»
    (r"soft margin", "sóft márgin"),
    (r"glm", "ge ele eme"), (r"penalty", "pénalti"), (r"None", "nan"),
    (r"VN", "uve ene"), (r"VP", "uve pe"), (r"w", "doble u"),
    (r"C", "ce"),                                                                             # (W) C -> «se», «el parámetro se»
]
PRON_C37 = PRON_C37_API + [p for p in R36.PRON_C36 if p[0] not in QUITAR] + PRON_C37_PAL

vp.NUM_DECIMAL = True
vp.SIMBOLOS_VOZ = True
vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + PRON_C37

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C37"
TMP = VIDEOS_DIR / "_tmp_c37_profe"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C37_01_ClasificarYLineaBase", "guion_c37_01_clasificar.json", "Introducción: clasificar y la línea base"),
    (2, "C37_02_MatrizConfusion", "guion_c37_02_matriz.json", "La matriz de confusión y las métricas"),
    (3, "C37_03_RegresionLogistica", "guion_c37_03_logistica.json", "Regresión logística: del puntaje a la probabilidad"),
    (4, "C37_04_UmbralYCoeficientes", "guion_c37_04_umbral.json", "Umbral, odds y coeficientes"),
    (5, "C37_05_MargenSVM", "guion_c37_05_margen.json", "SVM: hiperplano, margen y vectores de soporte"),
    (6, "C37_06_MargenSuave", "guion_c37_06_suave.json", "Margen suave, holgura y el parámetro C"),
    (7, "C37_07_Kernels", "guion_c37_07_kernels.json", "Kernels y el kernel radial a mano"),
    (8, "C37_08_CyGamma", "guion_c37_08_gamma.json", "C y gamma: subajuste y sobreajuste"),
    (9, "C37_09_Escalar", "guion_c37_09_escalar.json", "Estandarizar: SVM y kNN"),
    (10, "C37_10_Comparacion", "guion_c37_10_comparacion.json", "La comparación de los clasificadores"),
    (11, "C37_11_ValidacionAUC", "guion_c37_11_validacion.json", "Validación cruzada, búsqueda y AUC"),
    (12, "C37_12_Repaso", "guion_c37_12_repaso.json", "Repaso final y checklist"),
]
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def modo_base():
    """Linea base para medir la pronunciacion: mismas opciones del motor, pero solo las pronunciaciones de C35 y C36."""
    vp.PRONUNCIACION_EXTRA[:] = _DEFECTO + R36.PRON_C36


def construir_uno(n, solo_audio=False, tmp=None):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4" if tmp is None else Path(tmp) / f"{nombre}.mp4"   # --base no toca C37/
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
    meta = [";FFMETADATA1", "title=Clase 37 · SVM y clasificadores clásicos (completo)", "artist=Prof. Pedro Luiz Ramos"]
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
    srt = SALIDA / "C37_Completo_SVM.srt"
    srt.write_text("\n".join(srt_out), encoding="utf-8")
    (work / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    salida = SALIDA / "C37_Completo_SVM.mp4"
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
