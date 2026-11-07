"""Pipeline reutilizable para videos de estudio narrados (una clase = un guion JSON).

Uso desde un script de clase:
    import video_comun as vc
    vc.construir_video(guion_json, raiz_repo, salida_mp4, tmp_dir)

Formato del guion (ver guion_c30.json):
    {clase, titulo, subtitulo, profesora, pie, voz, rate,
     capitulos: [{titulo, escenas: [{kind?, title, bullets[], figure|null, narration}]}]}
    kind: "title" -> tarjeta de titulo; omitido -> diapositiva normal.

Requiere: edge-tts, pillow, ffmpeg/ffprobe en el PATH.
"""
import asyncio
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 10
PAD_INI = 0.5          # segundos de silencio antes de la narracion
PAD_FIN = 0.7          # segundos de silencio despues
PAUSA_FRASE = 0.28     # silencio entre frases
SR = 24000             # wav intermedio (mono, 16 bit)

# Paleta (coherente con las guias: azul marino + turquesa)
NAVY = (31, 58, 95)
TEAL = (14, 124, 134)
INK = (30, 41, 59)
MUTED = (100, 116, 139)
BG = (247, 249, 252)
CARD = (255, 255, 255)
LINE = (214, 222, 233)
TRACK = (221, 228, 238)

FONT_R = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"

# Pronunciacion solo para la voz (los subtitulos conservan el texto original).
PRONUNCIACION = [
    (r"\bone-hot\b", "uan jot"), (r"\bOne-hot\b", "Uan jot"),
    (r"\bRidge\b", "Ridch"), (r"\bLasso\b", "Laso"),
    (r"\brecall\b", "ríkol"), (r"\bRecall\b", "Ríkol"),
    (r"\bPipeline\b", "páiplain"), (r"\bK-Fold\b", "kei fold"),
    (r"\bscikit-learn\b", "saikit lern"), (r"\bF1\b", "efe uno"),
    (r"\bleakage\b", "líkech"), (r"\boverfitting\b", "overfíting"),
    (r"\bStratifiedKFold\b", "estratificado kei fold"),
    (r"\bColumnTransformer\b", "column transformer"),
    (r"\btrain test split\b", "train test split"),
    (r"\bfit\b", "fit"), (r"\bpandas\b", "pándas"),
    (r"\bPython\b", "Páizon"), (r"\baccuracy\b", "ácuresi"),
    (r"\bTP\b", "te pe"), (r"\bFP\b", "efe pe"), (r"\bFN\b", "efe ene"), (r"\bTN\b", "te ene"),
    (r"\bMAE\b", "eme a e"), (r"\bRMSE\b", "erre eme ese e"),
    (r"\bget dummies\b", "get dami"), (r"\bstratify\b", "estratifai"),
    (r"\bstandard\b", "estándar"),
]


def _fuente(ruta, tam):
    return ImageFont.truetype(ruta, tam)


def _ajustar(draw, texto, fuente, ancho):
    """Divide texto en lineas que caben en `ancho` px."""
    lineas, actual = [], ""
    for pal in texto.split():
        prueba = (actual + " " + pal).strip()
        if draw.textlength(prueba, font=fuente) <= ancho:
            actual = prueba
        else:
            if actual:
                lineas.append(actual)
            actual = pal
    if actual:
        lineas.append(actual)
    return lineas


def _bloque_bullets(draw, bullets, ancho, alto_max, tam_ini, tam_min=30):
    """Elige el mayor tamano de fuente con que los bullets caben en alto_max."""
    for tam in range(tam_ini, tam_min - 1, -2):
        f = _fuente(FONT_R, tam)
        sangria = int(tam * 1.0)
        inter = int(tam * 0.45)
        filas, y = [], 0
        for b in bullets:
            ls = _ajustar(draw, b, f, ancho - sangria)
            filas.append((y, ls))
            y += len(ls) * int(tam * 1.32) + inter
        if y - inter <= alto_max:
            return f, tam, sangria, filas, y - inter
    return f, tam, sangria, filas, y - inter


def _pie(img, d, guion, cap_i, n_caps, idx_global, total, cap_titulo):
    d.rectangle([0, H - 52, W, H], fill=CARD)
    d.line([0, H - 52, W, H - 52], fill=LINE, width=2)
    d.text((48, H - 44), guion["pie"], font=_fuente(FONT_R, 26), fill=MUTED)
    txt = f"{idx_global} / {total}"
    f = _fuente(FONT_B, 26)
    d.text((W - 48 - d.textlength(txt, font=f), H - 44), txt, font=f, fill=NAVY)
    # barra de progreso global
    bx0, bx1, by = 640, W - 240, H - 28
    d.rounded_rectangle([bx0, by - 6, bx1, by + 6], radius=6, fill=TRACK)
    fin = bx0 + int((bx1 - bx0) * idx_global / total)
    d.rounded_rectangle([bx0, by - 6, max(fin, bx0 + 12), by + 6], radius=6, fill=TEAL)


def render_diapositiva(guion, esc, cap_i, n_caps, idx_global, total, raiz, destino):
    cap_titulo = guion["capitulos"][cap_i]["titulo"]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    if esc.get("kind") == "title":
        img.paste(Image.new("RGB", (W, H), NAVY))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 24, H], fill=TEAL)
        d.text((140, 250), guion["clase"].upper(), font=_fuente(FONT_B, 44), fill=(150, 214, 220))
        ls = _ajustar(d, esc["title"], _fuente(FONT_B, 104), W - 300)
        y = 330
        for l in ls:
            d.text((140, y), l, font=_fuente(FONT_B, 104), fill=(255, 255, 255))
            y += 128
        y += 30
        d.rectangle([140, y, 420, y + 8], fill=TEAL)
        y += 56
        for b in esc["bullets"]:
            d.text((140, y), b, font=_fuente(FONT_R, 48), fill=(226, 234, 244))
            y += 70
        d.text((140, H - 110), "Diplomado Data Science UC 2026", font=_fuente(FONT_R, 32), fill=(150, 170, 200))
        img.save(destino)
        return

    # Banda de titulo
    d.rectangle([0, 0, W, 150], fill=NAVY)
    d.rectangle([0, 150, W, 156], fill=TEAL)
    etiqueta = f"CAPÍTULO {cap_i + 1} DE {n_caps}  ·  {cap_titulo.upper()}"
    d.text((64, 20), etiqueta, font=_fuente(FONT_B, 26), fill=(150, 214, 220))
    tf = _fuente(FONT_B, 60)
    t = esc["title"]
    while d.textlength(t, font=tf) > W - 128 and tf.size > 40:
        tf = _fuente(FONT_B, tf.size - 2)
    d.text((64, 62), t, font=tf, fill=(255, 255, 255))

    fig = esc.get("figure")
    zona_top, zona_bot = 190, H - 80
    if fig:
        # bullets arriba (max ~300 px), figura debajo
        f, tam, sang, filas, alto = _bloque_bullets(d, esc["bullets"], W - 160, 330, 44, 34)
        y0 = zona_top
        for y, ls in filas:
            yy = y0 + y
            d.ellipse([84, yy + tam * 0.48 - 8, 100, yy + tam * 0.48 + 8], fill=TEAL)
            for i, l in enumerate(ls):
                d.text((84 + sang, yy + i * int(tam * 1.32)), l, font=f, fill=INK)
        top_fig = y0 + alto + 28
        im = Image.open(Path(raiz) / fig).convert("RGB")
        aw, ah = W - 96, zona_bot - top_fig
        esc_f = min(aw / im.width, ah / im.height)
        nw, nh = int(im.width * esc_f), int(im.height * esc_f)
        im = im.resize((nw, nh), Image.LANCZOS)
        x = (W - nw) // 2
        y = top_fig + (ah - nh) // 2
        d.rounded_rectangle([x - 12, y - 12, x + nw + 12, y + nh + 12], radius=14, fill=CARD, outline=LINE, width=2)
        img.paste(im, (x, y))
    else:
        f, tam, sang, filas, alto = _bloque_bullets(d, esc["bullets"], W - 360, zona_bot - zona_top - 40, 58, 36)
        y0 = zona_top + max(20, (zona_bot - zona_top - alto) // 2 - 20)
        d.rounded_rectangle([80, y0 - 40, W - 80, y0 + alto + 40], radius=22, fill=CARD, outline=LINE, width=2)
        for y, ls in filas:
            yy = y0 + y
            d.ellipse([140, yy + tam * 0.48 - 10, 160, yy + tam * 0.48 + 10], fill=TEAL)
            for i, l in enumerate(ls):
                d.text((140 + sang + 8, yy + i * int(tam * 1.32)), l, font=f, fill=INK)
    _pie(img, d, guion, cap_i, n_caps, idx_global, total, cap_titulo)
    img.save(destino)


# ---------------------------------------------------------------- audio
def texto_voz(t):
    for pat, rep in PRONUNCIACION:
        t = re.sub(pat, rep, t)
    t = t.replace("²", " cuadrado").replace("%", " por ciento")
    return t


def frases(texto):
    partes = re.split(r"(?<=[.!?:])\s+", texto.strip())
    return [p for p in partes if p.strip()]


async def _sintetizar(texto, voz, rate, destino, sem):
    import edge_tts
    async with sem:
        for intento in range(4):
            try:
                await edge_tts.Communicate(texto, voz, rate=rate).save(str(destino))
                if destino.exists() and destino.stat().st_size > 800:
                    return
            except Exception:
                await asyncio.sleep(1.5 * (intento + 1))
        raise RuntimeError(f"TTS fallo: {texto[:60]}")


def _a_wav(mp3, wav):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR),
                    "-sample_fmt", "s16", str(wav)], check=True)


def _leer_pcm(wav):
    with wave.open(str(wav), "rb") as w:
        return w.readframes(w.getnframes())


def sintetizar_escenas(guion, cache):
    """Devuelve por escena: (pcm, [(frase_original, t_ini, t_fin)]) con audio cacheado por hash de frase."""
    cache.mkdir(parents=True, exist_ok=True)
    voz, rate = guion["voz"], guion["rate"]
    plan, tareas = [], []
    sem = asyncio.Semaphore(6)
    for cap in guion["capitulos"]:
        for esc in cap["escenas"]:
            fr = frases(esc["narration"])
            assert fr, f"narracion vacia: {esc['title']}"
            items = []
            for f in fr:
                tv = texto_voz(f)
                h = hashlib.sha1(f"{voz}|{rate}|{tv}".encode("utf-8")).hexdigest()[:20]
                mp3 = cache / f"{h}.mp3"
                if not mp3.exists():
                    tareas.append(_sintetizar(tv, voz, rate, mp3, sem))
                items.append((f, mp3, cache / f"{h}.wav"))
            plan.append(items)

    async def todo():
        await asyncio.gather(*tareas)
    if tareas:
        print(f"Sintetizando {len(tareas)} frases nuevas...")
        asyncio.run(todo())

    resultado = []
    pausa = b"\x00\x00" * int(SR * PAUSA_FRASE)
    for items in plan:
        pcm = b"\x00\x00" * int(SR * PAD_INI)
        cues = []
        for f, mp3, wav in items:
            if not wav.exists():
                _a_wav(mp3, wav)
            datos = _leer_pcm(wav)
            t0 = len(pcm) / 2 / SR
            pcm += datos
            cues.append((f, t0, len(pcm) / 2 / SR))
            pcm += pausa
        pcm += b"\x00\x00" * int(SR * (PAD_FIN - PAUSA_FRASE))
        # redondear la duracion a multiplo de 1/FPS
        n = len(pcm) // 2
        objetivo = -(-n * FPS // SR) * SR // FPS
        pcm += b"\x00\x00" * (objetivo - n)
        resultado.append((pcm, cues))
    return resultado


# ---------------------------------------------------------------- subtitulos
def _srt_t(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def _trozos(frase, maximo=84):
    """Parte una frase larga en trozos de <= maximo caracteres, cortando en espacios/comas."""
    if len(frase) <= maximo:
        return [frase]
    pals, trozos, act = frase.split(), [], ""
    n_trozos = -(-len(frase) // maximo)
    objetivo = len(frase) / n_trozos
    for p in pals:
        if act and len(act) + 1 + len(p) > objetivo * 1.12:
            trozos.append(act)
            act = p
        else:
            act = (act + " " + p).strip()
    if act:
        trozos.append(act)
    return trozos


def escribir_srt(cues_globales, destino):
    lineas, n = [], 1
    for frase, t0, t1 in cues_globales:
        tr = _trozos(frase)
        tot = sum(len(x) for x in tr)
        a = t0
        for x in tr:
            b = a + (t1 - t0) * len(x) / tot
            lineas += [str(n), f"{_srt_t(a)} --> {_srt_t(b)}", x, ""]
            n += 1
            a = b
    Path(destino).write_text("\n".join(lineas), encoding="utf-8")


# ---------------------------------------------------------------- ensamblado
def construir_video(guion_json, raiz, salida_mp4, tmp, conservar_cache=True):
    raiz, tmp = Path(raiz), Path(tmp)
    salida_mp4 = Path(salida_mp4)
    guion = json.loads(Path(guion_json).read_text(encoding="utf-8"))
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "slides").mkdir(exist_ok=True)

    audio = sintetizar_escenas(guion, tmp / "tts")
    escenas = [(ci, e) for ci, c in enumerate(guion["capitulos"]) for e in c["escenas"]]
    total, n_caps = len(escenas), len(guion["capitulos"])

    pcm_total, cues, lista, capitulos = b"", [], [], []
    t = 0.0
    cap_ini = {}
    informe = []
    for i, ((ci, esc), (pcm, cs)) in enumerate(zip(escenas, audio), start=1):
        png = tmp / "slides" / f"s{i:03}.png"
        render_diapositiva(guion, esc, ci, n_caps, i, total, raiz, png)
        dur = len(pcm) / 2 / SR
        cap_ini.setdefault(ci, t)
        cues += [(f, t + a, t + b) for f, a, b in cs]
        lista.append(f"file '{png.as_posix()}'\nduration {dur:.4f}")
        pcm_total += pcm
        informe.append((i, ci, esc["title"], t, dur))
        t += dur
    lista.append(f"file '{(tmp / 'slides' / f's{total:03}.png').as_posix()}'")
    (tmp / "lista.txt").write_text("\n".join(lista), encoding="utf-8")

    wav = tmp / "audio_total.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm_total)

    srt = salida_mp4.with_suffix(".srt")
    escribir_srt(cues, srt)

    meta = [";FFMETADATA1", f"title={guion['clase']} · {guion['titulo']}", "artist=" + guion.get("profesora", "")]
    inicios = sorted(cap_ini.items())
    for k, (ci, ini) in enumerate(inicios):
        fin = inicios[k + 1][1] if k + 1 < len(inicios) else t
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(ini * 1000)}", f"END={int(fin * 1000)}",
                 f"title={ci + 1}. {guion['capitulos'][ci]['titulo']}"]
    (tmp / "meta.txt").write_text("\n".join(meta), encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(tmp / "lista.txt"),
           "-i", str(wav), "-i", str(srt), "-i", str(tmp / "meta.txt"),
           "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3", "-map_chapters", "3",
           "-vf", f"fps={FPS},format=yuv420p", "-t", f"{t:.3f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "stillimage",
           "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-metadata:s:s:0", "language=spa",
           "-metadata:s:a:0", "language=spa", "-movflags", "+faststart", str(salida_mp4)]
    subprocess.run(cmd, check=True)
    print(f"OK {salida_mp4}  ({t / 60:.1f} min, {total} escenas, {n_caps} capítulos)")
    return informe, inicios, guion
