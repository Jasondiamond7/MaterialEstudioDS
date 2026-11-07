"""Clase 30 · serie completa de videos "profesor experto" (un video por seccion de la clase) + video completo.

Usa el motor video_profe.py y los guiones guion_c30_NN_*.json. El video 4 es el de muestra aprobado (se copia tal cual).

Ejecutar con el venv de Guias/videos/.venv:
    python video_c30_completo.py                       # construye los que falten + el video completo
    python video_c30_completo.py --solo 2 3            # solo esos videos (los reconstruye)
    python video_c30_completo.py --previa 2 12 40 95   # PNGs sueltos del video 2 en esos segundos
    python video_c30_completo.py --audio 2             # solo sintetiza audio / linea de tiempo del video 2
    python video_c30_completo.py --completo            # solo concatena el video completo
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

RAIZ = AQUI.parent.parent
VIDEOS_DIR = RAIZ / "Guias" / "videos"
SALIDA = VIDEOS_DIR / "C30"
TMP = VIDEOS_DIR / "_tmp_c30_profe"

# (numero, archivo de salida, guion, titulo de capitulo)
SERIE = [
    (1, "C30_01_Introduccion", "guion_c30_01_introduccion.json", "Introducción"),
    (2, "C30_02_TiposVariables", "guion_c30_02_tipos.json", "Tipos de variables"),
    (3, "C30_03_CalidadImputacion", "guion_c30_03_calidad.json", "Calidad de datos e imputación"),
    (4, "C30_04_EntrenamientoValidacionPrueba", None, "Entrenamiento, validación y prueba"),
    (5, "C30_05_Regularizacion", "guion_c30_05_regularizacion.json", "Regularización: Ridge, Lasso y estandarizar"),
    (6, "C30_06_KFold", "guion_c30_06_kfold.json", "Validación cruzada K-Fold"),
    (7, "C30_07_Estratificado", "guion_c30_07_estratificado.json", "Muestreo estratificado"),
    (8, "C30_08_MatrizConfusion", "guion_c30_08_matriz.json", "Matriz de confusión y métricas"),
    (9, "C30_09_FugaPipeline", "guion_c30_09_fuga.json", "Fuga de datos y Pipeline"),
    (10, "C30_10_Repaso", "guion_c30_10_repaso.json", "Repaso final"),
]
MUESTRA_MP4 = VIDEOS_DIR / "C30_Muestra_EntrenamientoValidacionPrueba.mp4"
MUESTRA_TMP = VIDEOS_DIR / "_tmp_c30_muestra"
MAX_CAP = 345.0          # un capitulo MP4 no debe superar ~340 s (desborde de ticks, ver video_profe.construir)


def _por_numero(n):
    return next(s for s in SERIE if s[0] == n)


def construir_uno(n, solo_audio=False):
    _, nombre, guion, _t = _por_numero(n)
    salida = SALIDA / f"{nombre}.mp4"
    SALIDA.mkdir(parents=True, exist_ok=True)
    if guion is None:                                   # video 4 = muestra aprobada, sin cambios
        shutil.copyfile(MUESTRA_MP4, salida)
        shutil.copyfile(MUESTRA_MP4.with_suffix(".srt"), salida.with_suffix(".srt"))
        print(f"OK {salida} (copia de la muestra aprobada)")
        return
    tmp = TMP / f"{n:02}"
    return vp.construir(AQUI / guion, salida, tmp, solo_audio=solo_audio)


def _dur(ruta, sel="v:0"):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "stream=duration",
                          "-of", "csv=p=0", str(ruta)], capture_output=True, text=True, check=True).stdout.strip()
    return float(out.split()[0])


def _secciones(n, total):
    """[(nombre, inicio)] de las secciones internas del video n (para sub-dividir capitulos largos)."""
    inf = (MUESTRA_TMP if n == 4 else TMP / f"{n:02}") / "informe.json"
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
    meta = [";FFMETADATA1", "title=Clase 30 · Preparación para Machine Learning (completo)", "artist=Prof. Laura Sánchez"]
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
        for k in range(1, partes):                       # cortes equilibrados, en limites de seccion interna
            objetivo = d * k / partes
            libres = [c for c in cand if c > cortes[-1] + 30]
            cortes.append(min(libres, key=lambda c: abs(c - objetivo)) if libres else objetivo)
        cortes.append(d)
        for j in range(len(cortes) - 1):
            nom = f"{n}. {titulo}" + (f" (parte {j + 1}/{len(cortes) - 1})" if len(cortes) > 2 else "")
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round((t0 + cortes[j]) * 1000))}",
                     f"END={int(round((t0 + cortes[j + 1]) * 1000))}", f"title={nom}"]
        t0 += d
    srt = SALIDA / "C30_Completo_PreparacionML.srt"
    srt.write_text("\n".join(srt_out), encoding="utf-8")
    (work / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    salida = SALIDA / "C30_Completo_PreparacionML.mp4"
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
