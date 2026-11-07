"""Clase 33 · serie completa de videos "profesor experto" (un video por seccion de la clase) + video completo.

Usa el motor video_profe.py y los guiones guion_c33_NN_*.json.
Ejecutar con el venv de Guias/videos/.venv:
    python video_c33_completo.py                       # construye los que falten + el video completo
    python video_c33_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c33_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c33_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c33_completo.py --completo            # solo concatena el video completo
"""
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import video_profe as vp  # noqa: E402

# Pronunciaciones propias de la Clase 33 (se anaden solo en este proceso; el motor y las otras clases no cambian).
# Se aplican en orden, despues de convertir los numeros a palabras.
vp.PRONUNCIACION_EXTRA.extend([
    (r"_", " "),                                    # nombres de variables: log_segundos -> log segundos
    (r"\bn contactos\b", "número de contactos"), (r"\bfrec\b", "frecuencia"), (r"\bsms\b", "ese eme ese"),
    (r"\bclv\b", "ce ele ve"), (r"\bCLV\b", "ce ele ve"), (r"\bSMS\b", "ese eme ese"),
    (r"\bq(\d)\b", lambda m: "cu " + vp.entero_a_palabras(int(m.group(1)))),
    (r"\bx(\d)\b", lambda m: "equis " + vp.entero_a_palabras(int(m.group(1)))),
    (r"\bAUC\b", "a u ce"), (r"\bXOR\b", "équis or"), (r"\bRFECV\b", "erre efe e ce ve"), (r"\bRFE\b", "erre efe e"),
    (r"\bANOVA\b", "anóva"), (r"\bSelectKBest\b", "sélect kei best"), (r"\bVarianceThreshold\b", "veriáns thréshold"),
    (r"\bthreshold\b", "thréshold"), (r"\bliblinear\b", "lib líniar"), (r"\bsolver\b", "sólver"),
    (r"\brandom state\b", "rándom estéit"), (r"\bRandom Forest\b", "rándom fórest"), (r"\bsklearn\b", "saikit lern"),
    (r"\bcross val score\b", "cros val escór"), (r"\bpipeline\b", "páiplain"), (r"\bwrapper\b", "uráper"),
    (r"\bWrapper\b", "Uráper"), (r"\bembedded\b", "embéded"), (r"\bEmbedded\b", "Embéded"), (r"\bfilter\b", "fílter"),
    (r"\bFilter\b", "Fílter"), (r"\bfeature engineering\b", "fíchur enyiníring"),
    (r"\bFeature engineering\b", "Fíchur enyiníring"), (r"\bchurn\b", "chern"), (r"\bChurn\b", "Chern"),
    (r"\btarget\b", "tárget"), (r"\bdataset\b", "dáta set"), (r"\bfolds?\b", "fóld"),
    (r"\bp-valor(es)?\b", lambda m: "pe valor" + (m.group(1) or "")),
    (r"\bWeight of Evidence\b", "uéit of évidens"), (r"\bInformation Value\b", "informéishon váliu"), (r"\bWOE\b", "wóu"),
    (r"\bPCA\b", "pe ce a"), (r"\bML\b", "eme ele"), (r"\btelco\b", "télco"), (r"\bL1\b", "ele uno"), (r"\bL2\b", "ele dos"),
    (r"\bNaN\b", "nan"), (r"\bscore\b", "escór"), (r"α", "alfa"), (r"π", "pi"), (r"\btop\b", "tóp"),
    (r"\bdescribe\b", "discráib"), (r"\bratio\b", "rasio"), (r"\bratios\b", "rasios"),
])

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C33"
TMP = VIDEOS_DIR / "_tmp_c33_profe"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C33_01_Introduccion", "guion_c33_01_introduccion.json", "Introducción: ¿con cuáles variables me quedo?"),
    (2, "C33_02_LimpiarConCriterio", "guion_c33_02_limpiar.json", "Limpiar con criterio de negocio"),
    (3, "C33_03_ConstruirVariables", "guion_c33_03_construir.json", "Construir variables: de 12 a 50"),
    (4, "C33_04_IrrelevanteRedundante", "guion_c33_04_motivos.json", "Irrelevante, redundante o útil en conjunto"),
    (5, "C33_05_CasoXOR", "guion_c33_05_xor.json", "El caso XOR: útil solo en conjunto"),
    (6, "C33_06_MetodosFiltro", "guion_c33_06_filtros.json", "Taxonomía y métodos de filtro"),
    (7, "C33_07_CuatroContraCuatro", "guion_c33_07_topk.json", "Cuatro contra cuatro: el ranking ingenuo"),
    (8, "C33_08_WrapperEmbedded", "guion_c33_08_wrapper.json", "Wrapper y embedded: RFE y Lasso"),
    (9, "C33_09_LaTrampa", "guion_c33_09_trampa.json", "La trampa: dónde vive la selección"),
    (10, "C33_10_Repaso", "guion_c33_10_repaso.json", "Repaso final y checklist"),
]
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def construir_uno(n, solo_audio=False):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4"
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = TMP / f"{n:02}"
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
    # 1) video: concat (copia)
    (work / "lista.txt").write_text("\n".join(f"file '{a.as_posix()}'" for a in archivos), encoding="utf-8")
    video = work / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "lista.txt"),
                    "-map", "0:v", "-c", "copy", "-video_track_timescale", "1000", str(video)], check=True)
    # 2) audio: decodifica cada uno, ajusta a la duracion de su video y une (una sola codificacion AAC)
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
    # 3) subtitulos con desfase + capitulos
    srt_out, k, t0 = [], 1, 0.0
    meta = [";FFMETADATA1", "title=Clase 33 · Feature engineering: metodología de selección de variables (completo)", "artist=Prof. Eduardo Rodríguez"]
    for (n, nombre, _g, titulo), a, d in zip(SERIE, archivos, durs):
        for bloque in a.with_suffix(".srt").read_text(encoding="utf-8").strip().split("\n\n"):
            ls = bloque.strip().split("\n")
            if len(ls) < 3:
                continue
            ini, fin = ls[1].split(" --> ")
            srt_out.append(f"{k}\n{_srt_f(t0 + _srt_t(ini))} --> {_srt_f(t0 + _srt_t(fin))}\n" + "\n".join(ls[2:]) + "\n")
            k += 1
        # capitulos: uno por video; si dura > MAX_CAP se divide en cortes de seccion interna
        secs = _secciones(n, d)
        cand = [s_ for _n, s_ in secs[1:]]
        partes = max(1, math.ceil(d / MAX_CAP))
        cortes = [0.0]
        for kk in range(1, partes):                       # cortes equilibrados, en limites de seccion interna
            objetivo = d * kk / partes
            libres = [c for c in cand if c > cortes[-1] + 30]
            cortes.append(min(libres, key=lambda c: abs(c - objetivo)) if libres else objetivo)
        cortes.append(d)
        for j in range(len(cortes) - 1):
            nom = f"{n}. {titulo}" + (f" (parte {j + 1}/{len(cortes) - 1})" if len(cortes) > 2 else "")
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round((t0 + cortes[j]) * 1000))}",
                     f"END={int(round((t0 + cortes[j + 1]) * 1000))}", f"title={nom}"]
        t0 += d
    srt = SALIDA / "C33_Completo_FeatureEngineering.srt"
    srt.write_text("\n".join(srt_out), encoding="utf-8")
    (work / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    salida = SALIDA / "C33_Completo_FeatureEngineering.mp4"
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
