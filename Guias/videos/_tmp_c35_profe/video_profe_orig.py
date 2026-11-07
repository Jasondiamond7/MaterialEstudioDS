"""Motor de videos "profesor experto": narracion por beats + animaciones sincronizadas con la voz.

Reutilizable para cualquier capitulo de cualquier clase: el contenido vive en un guion JSON y este
modulo hace TTS, sincronia, animacion, subtitulos y ensamblado.

Uso:
    import video_profe as vp
    vp.construir(guion_json, salida_mp4, tmp_dir)

Guion (ver guion_c30_muestra.json):
    {clase, titulo, voz, rate, fps?, cabecera?,
     beats: [{seccion?, say, pausa?, acciones: [...]}]}

say: texto que se ve en subtitulos. Marcas:
    [[a]]           punto de sincronia: una accion con "en": "a" arranca cuando la voz dice la palabra siguiente.
    {210|doscientas diez}  se muestra "210" y la voz dice "doscientas diez" (genero, siglas...).
    Los numeros sueltos se leen solos: 0.999 -> "cero coma novecientos noventa y nueve", 1.000 -> "mil".

acciones (se ejecutan al inicio del beat, o en la marca "en", mas "t" segundos opcionales):
    {"id": "x", "tipo": "texto", ...props, "entrada": "fade|subir|pop|ninguna", "dur": 0.6}  crea elemento
    {"id": "x", ...props, "dur": 0.6, "ease": "inout|out|back|lineal"}  anima props (rutas "series.0.revela")
    {"quitar": ["x", "y"]}            desvanece y elimina
    {"limpiar": true, "mantener": [...]}  desvanece todo salvo lo indicado

Tipos de elemento: texto, numero, caja, anillo, linea, puntos, grafico, candado, marca, tarjeta.
Coordenadas en px de un lienzo 1920x1080; con "ref": "<id grafico>" se usan coordenadas de datos.

Requiere: edge-tts, pillow, numpy, ffmpeg/ffprobe en el PATH.
"""
import asyncio
import bisect
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import video_comun as vc  # noqa: E402  (TTS, pronunciacion y SRT reutilizados, solo lectura)

W, H = 1920, 1080
FPS = 25
SR = 24000
SS = 2                    # supersampling para bordes suaves
GAP_FRASE = 0.12          # silencio insertado entre frases (y entre beats)
MARGEN_INI = 0.04         # audio conservado antes de la primera palabra
MARGEN_FIN = 0.10         # audio conservado tras la ultima palabra
PAUSA_MAX = 0.42          # silencios internos mas largos se recortan ...
PAUSA_OBJ = 0.30          # ... a esta duracion
COLA = 0.5                # cierre tras la ultima frase
ANTICIPO = 0.08           # la animacion de una marca arranca un poco antes de la palabra

FONT_R = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"
FONT_SB = "C:/Windows/Fonts/seguisb.ttf"
FONT_M = "C:/Windows/Fonts/consola.ttf"
FONTS = {"r": FONT_R, "b": FONT_B, "sb": FONT_SB, "m": FONT_M}

PALETA = {
    "fondo": (247, 249, 252), "blanco": (255, 255, 255), "tinta": (30, 41, 59), "gris": (100, 116, 139),
    "grisclaro": (203, 213, 225), "linea": (214, 222, 233), "navy": (31, 58, 95), "teal": (14, 124, 134),
    "violeta": (124, 58, 237), "azul": (37, 99, 235), "rojo": (220, 38, 38), "verde": (5, 150, 105),
    "ambar": (217, 119, 6), "violetaclaro": (237, 233, 254), "azulclaro": (219, 234, 254),
    "rojoclaro": (254, 226, 226), "verdeclaro": (209, 250, 229), "ambarclaro": (254, 243, 199),
    "tealclaro": (204, 251, 241),
}


# ================================================================ numeros a voz
_UNI = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
        "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno",
        "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
        "veintinueve"]
_DEC = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_CEN = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos", 6: "seiscientos",
        7: "setecientos", 8: "ochocientos", 9: "novecientos"}


def entero_a_palabras(n):
    if n < 30:
        return _UNI[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _DEC[d] + ("" if u == 0 else " y " + _UNI[u])
    if n < 1000:
        c, r = divmod(n, 100)
        if c == 1 and r == 0:
            return "cien"
        return _CEN[c] + ("" if r == 0 else " " + entero_a_palabras(r))
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        base = "mil" if m == 1 else re.sub(r"uno$", "un", entero_a_palabras(m)).replace("veintiun", "veintiún") + " mil"
        return base + ("" if r == 0 else " " + entero_a_palabras(r))
    m, r = divmod(n, 1_000_000)
    base = "un millón" if m == 1 else re.sub(r"uno$", "un", entero_a_palabras(m)) + " millones"
    return base + ("" if r == 0 else " " + entero_a_palabras(r))


def numero_a_palabras(tok):
    """'0.999' -> cero coma novecientos noventa y nueve; '1.000' -> mil; '0,04' -> cero coma cero cuatro."""
    if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+", tok):
        return entero_a_palabras(int(tok.replace(".", "")))
    m = re.fullmatch(r"(\d+)[.,](\d+)", tok)
    if m:
        ent, dec = m.groups()
        ceros = len(dec) - len(dec.lstrip("0"))
        resto = dec.lstrip("0")
        partes = ["cero"] * ceros + ([entero_a_palabras(int(resto))] if resto else [])
        return entero_a_palabras(int(ent)) + " coma " + " ".join(partes)
    return entero_a_palabras(int(tok))


_RE_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+[.,]\d+|\d+)(?![\w])")
_RE_MARKUP = re.compile(r"\{([^{}|]*)\|([^{}]*)\}")
_RE_MARCA = re.compile(r"\[\[(\w+)\]\]")


# Pronunciaciones adicionales (anglicismos que el TTS en espanol lee mal). Solo anaden; no alteran lo existente.
PRONUNCIACION_EXTRA = [
    (r"predict", "predíct"), (r"transform", "transfórm"), (r"fit_transform", "fit transfórm"),
    (r"StandardScaler", "estándar skéiler"), (r"scaler", "skéiler"), (r"Elastic Net", "elástic net"),
    (r"SimpleImputer", "símple impúter"), (r"data leakage", "dáta líkech"),
    (r"Shuffle", "shófol"), (r"PR-AUC", "pe erre a u ce"),
]


def a_pantalla(t):
    """Texto de subtitulo: sin marcas, markup -> parte visible."""
    return re.sub(r"\s{2,}", " ", _RE_MARKUP.sub(lambda m: m.group(1), _RE_MARCA.sub("", t))).strip()


def a_voz(t):
    """Texto para el TTS: markup -> parte hablada, numeros -> palabras, pronunciacion de video_comun."""
    t = _RE_MARCA.sub("", t)
    guard = []

    def prot(m):
        guard.append(m.group(2))
        return chr(0xE100 + len(guard) - 1)     # marcador sin digitos (no lo toca _RE_NUM)
    t = _RE_MARKUP.sub(prot, t)
    t = t.replace("R²", "R cuadrado")
    t = _RE_NUM.sub(lambda m: numero_a_palabras(m.group(1)), t)
    t = re.sub("[-]", lambda m: guard[ord(m.group(0)) - 0xE100], t)
    t = vc.texto_voz(t)
    for pat, rep in PRONUNCIACION_EXTRA:
        t = re.sub(pat, rep, t)
    t = t.replace("−", " menos ").replace("×", " por ")
    return re.sub(r"\s{2,}", " ", t).strip()


def frases(texto):
    return [p for p in re.split(r"(?<=[.!?…])\s+", texto.strip()) if p.strip()]


def _norm(w):
    import unicodedata
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if c.isalnum() and unicodedata.category(c) != "Mn")


def _palabras(t):
    return [x for x in (_norm(p) for p in re.findall(r"[\w²]+", t)) if x]


# ================================================================ TTS con limites de palabra
async def _tts_palabras(texto, voz, rate, mp3, js, sem):
    import edge_tts
    async with sem:
        for intento in range(5):
            try:
                c = edge_tts.Communicate(texto, voz, rate=rate, boundary="WordBoundary")
                audio, ev = bytearray(), []
                async for ch in c.stream():
                    if ch["type"] == "audio":
                        audio += ch["data"]
                    elif ch["type"] == "WordBoundary":
                        ev.append([ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]])
                if len(audio) > 800 and ev:
                    mp3.write_bytes(bytes(audio))
                    js.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
                    return
            except Exception:
                pass
            await asyncio.sleep(1.5 * (intento + 1))
        raise RuntimeError(f"TTS fallo: {texto[:70]}")


def _recortar(pcm, palabras):
    """Recorta silencio inicial/final y acorta pausas internas largas. Devuelve (pcm, palabras ajustadas)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    ventana = int(SR * 0.01)
    n = len(x) // ventana
    rms = np.sqrt((x[: n * ventana].reshape(n, ventana) ** 2).mean(axis=1) + 1e-9)
    voz = rms > 32768 * 10 ** (-42 / 20)
    if not voz.any():
        return pcm, palabras
    i0 = max(0, int(np.argmax(voz)) * ventana - int(SR * MARGEN_INI))
    i1 = min(len(x), (n - int(np.argmax(voz[::-1]))) * ventana + int(SR * MARGEN_FIN))
    cortes = []                                     # (inicio_muestra, muestras_quitadas) en coords originales
    run = None
    for k in range(int(np.argmax(voz)), n - int(np.argmax(voz[::-1]))):
        if not voz[k]:
            run = k if run is None else run
        elif run is not None:
            largo = (k - run) * ventana
            if largo > SR * PAUSA_MAX:
                quitar = largo - int(SR * PAUSA_OBJ)
                cortes.append((run * ventana + int(SR * PAUSA_OBJ / 2), quitar))
            run = None
    piezas, pos = [], i0
    for ini, q in cortes:
        piezas.append(x[pos:ini])
        pos = ini + q
    piezas.append(x[pos:i1])
    y = np.concatenate(piezas).astype(np.int16)

    def mover(t):
        s = t * SR
        quitado = sum(q for ini, q in cortes if ini < s)
        return max(0.0, (s - i0 - quitado) / SR)
    nuevas = [[mover(o), d, w] for o, d, w in palabras]
    return y.tobytes(), nuevas


def sintetizar(guion, cache):
    """Por beat: lista de frases {texto, voz, pcm, palabras, marcas{nombre: idx_palabra}}."""
    cache.mkdir(parents=True, exist_ok=True)
    voz, rate = guion["voz"], guion["rate"]
    sem = asyncio.Semaphore(6)
    tareas, plan = [], []
    for b in guion["beats"]:
        fr_plan = []
        for f in frases(b["say"]):
            hablado = a_voz(f)
            h = hashlib.sha1(f"{voz}|{rate}|{hablado}".encode("utf-8")).hexdigest()[:20]
            mp3, js = cache / f"{h}.mp3", cache / f"{h}.json"
            if not (mp3.exists() and js.exists()):
                tareas.append(_tts_palabras(hablado, voz, rate, mp3, js, sem))
            marcas = {}
            for m in _RE_MARCA.finditer(f):
                previo = _palabras(a_voz(f[: m.start()]))
                siguiente = _palabras(a_voz(_RE_MARCA.sub("", f[m.end():])))
                marcas[m.group(1)] = (len(previo), siguiente[0] if siguiente else "")
            fr_plan.append(dict(texto=a_pantalla(f), voz=hablado, mp3=mp3, js=js, wav=cache / f"{h}.wav", marcas=marcas))
        plan.append(fr_plan)
    if tareas:
        print(f"Sintetizando {len(tareas)} frases nuevas...")

        async def todo():
            await asyncio.gather(*tareas)
        asyncio.run(todo())
    for fr_plan in plan:
        for fr in fr_plan:
            if not fr["wav"].exists():
                vc._a_wav(fr["mp3"], fr["wav"])
            pal = json.loads(fr["js"].read_text(encoding="utf-8"))
            fr["pcm"], fr["palabras"] = _recortar(vc._leer_pcm(fr["wav"]), pal)
    return plan


# ================================================================ interpolacion
def ease(nombre, p):
    p = min(1.0, max(0.0, p))
    if nombre == "lineal":
        return p
    if nombre == "out":
        return 1 - (1 - p) ** 3
    if nombre == "in":
        return p ** 3
    if nombre == "back":
        c = 1.70158
        return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def mezclar(a, b, p):
    if isinstance(b, bool) or isinstance(a, bool):
        return b if p >= 0.5 else a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * p
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return type(b)(mezclar(x, y, p) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: mezclar(a.get(k, v), v, p) for k, v in b.items()}
    return b if p >= 0.5 else a


def color(c):
    if c is None or isinstance(c, tuple):
        return c
    if isinstance(c, list):
        return tuple(c)
    if c.startswith("#"):
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    return PALETA[c]


def _es_color(clave):
    return "color" in clave or clave in ("relleno", "borde", "fondo_txt")


def _resolver_colores(v, clave=""):
    if isinstance(v, dict):
        return {k: _resolver_colores(x, k) for k, x in v.items()}
    es_rgb = isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(z, (int, float)) for z in v)
    if _es_color(clave) and (es_rgb or isinstance(v, str) or v is None):
        return color(v)
    if isinstance(v, list):
        return [_resolver_colores(x, clave) for x in v]
    return v


def _set_ruta(d, ruta, val):
    partes = ruta.split(".")
    cur = d
    for p in partes[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    ult = partes[-1]
    if isinstance(cur, list):
        cur[int(ult)] = val
    else:
        cur[ult] = val


def _ci(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]))


@lru_cache(maxsize=512)
def fuente(peso, tam):
    return ImageFont.truetype(FONTS.get(peso, FONT_R), max(4, int(round(tam))))


_MEDIDOR = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def _lienzo(frame, bbox, alfa, fn, aa=True):
    """Dibuja con fn(d, X, Y, k) sobre la region bbox (supersampleada) y la mezcla con alfa."""
    x0, y0, x1, y1 = (int(math.floor(bbox[0])), int(math.floor(bbox[1])), int(math.ceil(bbox[2])), int(math.ceil(bbox[3])))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 1 or y1 - y0 < 1 or alfa <= 0.003:
        return
    region = frame.crop((x0, y0, x1, y1))
    k = SS if aa else 1
    big = region.resize(((x1 - x0) * k, (y1 - y0) * k), Image.NEAREST) if k > 1 else region.copy()
    d = ImageDraw.Draw(big)
    fn(d, lambda x: (x - x0) * k, lambda y: (y - y0) * k, k)
    small = big.reduce(k) if k > 1 else big
    if alfa < 0.997:
        small = Image.blend(region, small, alfa)
    frame.paste(small, (x0, y0))


# ================================================================ elementos
class Elemento:
    base = {}
    vivo = False          # anima por si solo (frame siempre "sucio")

    def preparar(self, p):
        return p

    def mezclar(self, a, b, p_lin, nombre_ease):
        return mezclar(a, b, ease(nombre_ease, p_lin))

    def es_vivo(self, p):
        return False

    def dibujar(self, frame, p, ctx):
        raise NotImplementedError


def _texto_lineas(texto, f, ancho_max):
    if not ancho_max:
        return texto.split("\n")
    out = []
    for parrafo in texto.split("\n"):
        out += vc._ajustar(_MEDIDOR, parrafo, f, ancho_max) or [""]
    return out


class Texto(Elemento):
    base = dict(x=960, y=540, texto="", tam=48, peso="r", color="tinta", ancla="mm", alfa=1.0, escala=1.0,
                dx=0.0, dy=0.0, ancho_max=0, interlinea=1.22, fondo_txt=None, pad=18, tachado=0.0)

    def contenido(self, p):
        return p["texto"]

    def dibujar(self, frame, p, ctx):
        txt = self.contenido(p)
        if not txt:
            return
        tam = p["tam"] * p["escala"]
        f = fuente(p["peso"], tam)
        lineas = _texto_lineas(txt, f, p["ancho_max"] * p["escala"] if p["ancho_max"] else 0)
        alto_l = tam * p["interlinea"]
        anchos = [_MEDIDOR.textlength(l, font=f) for l in lineas]
        aw, ah = max(anchos), alto_l * len(lineas)
        ha, va = p["ancla"][0], p["ancla"][1]
        cx = p["x"] + p["dx"]
        cy = p["y"] + p["dy"]
        left = cx - (aw / 2 if ha == "m" else (aw if ha == "r" else 0))
        top = cy - (ah / 2 if va == "m" else (ah if va in "bds" else 0))
        pad = p["pad"] * p["escala"] if p["fondo_txt"] else 4
        bbox = (left - pad - 6, top - pad - 6, left + aw + pad + 6, top + ah + pad + 6)

        def fn(d, X, Y, k):
            if p["fondo_txt"]:
                d.rounded_rectangle([X(left - pad), Y(top - pad * 0.6), X(left + aw + pad), Y(top + ah + pad * 0.6)],
                                    radius=int((ah + pad) * 0.5 * k), fill=_ci(p["fondo_txt"]))
            fk = fuente(p["peso"], tam * k)
            for i, (l, a) in enumerate(zip(lineas, anchos)):
                lx = left + (aw - a) / 2 if ha == "m" else (left + aw - a if ha == "r" else left)
                ly = top + i * alto_l + alto_l / 2
                d.text((X(lx), Y(ly)), l, font=fk, fill=_ci(p["color"]), anchor="lm")
            if p["tachado"] > 0:
                ty = top + ah / 2
                d.line([X(left - 8), Y(ty), X(left - 8 + (aw + 16) * p["tachado"]), Y(ty)], fill=_ci(PALETA["rojo"]),
                       width=int(6 * k))
        _lienzo(frame, bbox, p["alfa"], fn)


class Numero(Texto):
    base = dict(Texto.base, valor=0.0, fmt="{:.0f}", prefijo="", sufijo="", peso="b")

    def contenido(self, p):
        return p["prefijo"] + p["fmt"].format(p["valor"]) + p["sufijo"]


class Caja(Elemento):
    """Rectangulo redondeado. Centro (x,y,w,h) o esquinas (x0,y0,x1,y1, utiles con ref)."""
    base = dict(x=960, y=540, w=200, h=100, radio=18, relleno="blanco", borde=None, grosor=3, alfa=1.0,
                escala=1.0, guion=False, sombra=False, dy=0.0)

    def preparar(self, p):
        if "x0" in p:
            p["x"], p["y"] = (p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2
            p["w"], p["h"] = abs(p["x1"] - p["x0"]), abs(p["y1"] - p["y0"])
        return p

    def dibujar(self, frame, p, ctx):
        if "x0" in p:
            p = self.preparar(dict(p))
        w, h = p["w"] * p["escala"], p["h"] * p["escala"]
        if w < 2 or h < 2:
            return
        x0, y0 = p["x"] - w / 2, p["y"] + p["dy"] - h / 2
        x1, y1 = x0 + w, y0 + h
        r = max(0, min(p["radio"] * p["escala"], w / 2 - 1, h / 2 - 1))
        m = p["grosor"] + 14

        def fn(d, X, Y, k):
            if p["sombra"]:
                d.rounded_rectangle([X(x0 + 2), Y(y0 + 6), X(x1 + 2), Y(y1 + 6)], radius=int(r * k),
                                    fill=(226, 232, 240))
            if p["relleno"] is not None:
                d.rounded_rectangle([X(x0), Y(y0), X(x1), Y(y1)], radius=int(r * k), fill=_ci(p["relleno"]))
            if p["borde"] is not None and p["grosor"] > 0:
                g = max(1, int(p["grosor"] * k))
                if p["guion"]:
                    _rect_guion(d, X(x0), Y(y0), X(x1), Y(y1), _ci(p["borde"]), g, 18 * k)
                else:
                    d.rounded_rectangle([X(x0), Y(y0), X(x1), Y(y1)], radius=int(r * k), outline=_ci(p["borde"]), width=g)
        _lienzo(frame, (x0 - m, y0 - m, x1 + m, y1 + m + 6), p["alfa"], fn, aa=p["radio"] > 0 or p["borde"] is not None)


def _rect_guion(d, x0, y0, x1, y1, c, g, paso):
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        largo = math.hypot(bx - ax, by - ay)
        n = max(1, int(largo / paso))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1, (i + 1) / n)
            d.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1], fill=c, width=g)


class Anillo(Elemento):
    """Resaltador: elipse o rectangulo redondeado (radio>0) con pulso opcional."""
    base = dict(x=960, y=540, w=200, h=120, radio=0, color="ambar", grosor=6, alfa=1.0, escala=1.0, pulso=True,
                progreso=1.0)

    def es_vivo(self, p):
        return p["pulso"] and p["alfa"] > 0.01

    def dibujar(self, frame, p, ctx):
        s = p["escala"] * (1 + (0.035 * math.sin(ctx["t"] * 2 * math.pi * 1.1) if p["pulso"] else 0))
        w, h = p["w"] * s, p["h"] * s
        x0, y0, x1, y1 = p["x"] - w / 2, p["y"] - h / 2, p["x"] + w / 2, p["y"] + h / 2
        g = p["grosor"]

        def fn(d, X, Y, k):
            if p["radio"] > 0:
                d.rounded_rectangle([X(x0), Y(y0), X(x1), Y(y1)], radius=int(p["radio"] * k), outline=_ci(p["color"]),
                                    width=int(g * k))
            else:
                ang = 360 * p["progreso"]
                d.arc([X(x0), Y(y0), X(x1), Y(y1)], -90, -90 + ang, fill=_ci(p["color"]), width=int(g * k))
        _lienzo(frame, (x0 - g - 4, y0 - g - 4, x1 + g + 4, y1 + g + 4), p["alfa"], fn)


class Linea(Elemento):
    base = dict(x0=0, y0=0, x1=100, y1=100, progreso=1.0, color="gris", grosor=5, punta=False, guion=False, alfa=1.0)

    def dibujar(self, frame, p, ctx):
        pr = max(0.0, min(1.0, p["progreso"]))
        if pr <= 0.001:
            return
        ax, ay = p["x0"], p["y0"]
        bx, by = ax + (p["x1"] - ax) * pr, ay + (p["y1"] - ay) * pr
        g = p["grosor"]
        m = g * 4 + 10

        def fn(d, X, Y, k):
            c = _ci(p["color"])
            ex, ey = bx, by
            if p["punta"]:
                ang = math.atan2(by - ay, bx - ax)
                L = g * 3.6
                ex, ey = bx - math.cos(ang) * L * 0.8, by - math.sin(ang) * L * 0.8
            if p["guion"]:
                largo = math.hypot(ex - ax, ey - ay)
                n = max(1, int(largo / 22))
                for i in range(0, n, 2):
                    t0, t1 = i / n, min(1, (i + 1) / n)
                    d.line([X(ax + (ex - ax) * t0), Y(ay + (ey - ay) * t0), X(ax + (ex - ax) * t1), Y(ay + (ey - ay) * t1)],
                           fill=c, width=int(g * k))
            else:
                d.line([X(ax), Y(ay), X(ex), Y(ey)], fill=c, width=int(g * k))
            if p["punta"]:
                ang = math.atan2(by - ay, bx - ax)
                L, A = g * 3.6, 0.5
                pts = [(bx, by), (bx - L * math.cos(ang - A), by - L * math.sin(ang - A)),
                       (bx - L * math.cos(ang + A), by - L * math.sin(ang + A))]
                d.polygon([(X(x), Y(y)) for x, y in pts], fill=c)
        _lienzo(frame, (min(ax, bx) - m, min(ay, by) - m, max(ax, bx) + m, max(ay, by) + m), p["alfa"], fn)


def _rejilla(n, rect, radio):
    """Posiciones de n puntos ordenados en una rejilla que llena rect=[x0,y0,x1,y1]."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    cols = max(1, math.ceil(math.sqrt(n * w / max(h, 1))))
    filas = math.ceil(n / cols)
    while filas > 1 and (filas - 1) * cols >= n:
        filas -= 1
    paso = min(w / cols, h / max(filas, 1))
    ancho, alto = paso * (cols - 1), paso * (filas - 1)
    ox, oy = x0 + (w - ancho) / 2, y0 + (h - alto) / 2
    return [[ox + (i % cols) * paso, oy + (i // cols) * paso] for i in range(n)]


class Puntos(Elemento):
    """n puntos repartidos en grupos: grupos=[{n, rect, color}], visibles = cuantos se ven (onda de aparicion).
    Al cambiar grupos los puntos viajan con escalonado (escalonar 0..0.8)."""
    base = dict(grupos=[], radio=11, visibles=1e9, alfa=1.0, escalonar=0.5, resaltar=-1, atenuar=1.0)

    def preparar(self, p):
        pos, cols = [], []
        for g in p["grupos"]:
            pos += _rejilla(int(g["n"]), g["rect"], p["radio"])
            cols += [color(g["color"])] * int(g["n"])
        p["pos"], p["cols"] = pos, cols
        return p

    def mezclar(self, a, b, p_lin, nombre_ease):
        out = mezclar({k: v for k, v in a.items() if k not in ("pos", "cols")},
                      {k: v for k, v in b.items() if k not in ("pos", "cols")}, ease(nombre_ease, p_lin))
        n = len(b["pos"])
        if len(a["pos"]) != n:
            out["pos"], out["cols"] = b["pos"], b["cols"]
            return out
        s = b.get("escalonar", 0.5)
        pos, cols = [], []
        for i in range(n):
            pi = ease(nombre_ease, (p_lin - s * i / max(n - 1, 1)) / (1 - s)) if s < 1 else ease(nombre_ease, p_lin)
            pa, pb = a["pos"][i], b["pos"][i]
            pos.append([pa[0] + (pb[0] - pa[0]) * pi, pa[1] + (pb[1] - pa[1]) * pi])
            ca, cb = a["cols"][i], b["cols"][i]
            cols.append(tuple(ca[j] + (cb[j] - ca[j]) * pi for j in range(3)))
        out["pos"], out["cols"] = pos, cols
        return out

    def dibujar(self, frame, p, ctx):
        pos = p["pos"]
        if not pos:
            return
        r = p["radio"]
        xs, ys = [q[0] for q in pos], [q[1] for q in pos]
        vis = p["visibles"]

        def fn(d, X, Y, k):
            for i, (q, c) in enumerate(zip(pos, p["cols"])):
                if i >= vis:
                    break
                f = min(1.0, vis - i)          # el ultimo punto visible crece de a poco
                rr = r * (0.3 + 0.7 * f)
                cc = c if p["atenuar"] >= 0.999 else tuple(255 - (255 - v) * p["atenuar"] for v in c)
                d.ellipse([X(q[0] - rr), Y(q[1] - rr), X(q[0] + rr), Y(q[1] + rr)], fill=_ci(cc))
        _lienzo(frame, (min(xs) - r - 4, min(ys) - r - 4, max(xs) + r + 4, max(ys) + r + 4), p["alfa"], fn)


def _catmull(pts, pasos=14):
    if len(pts) < 3:
        return pts
    out = []
    ext = [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(pasos):
            t = s / pasos
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    out.append(pts[-1])
    return out


class Grafico(Elemento):
    """Ejes + series que se dibujan solas (revela 0..1 a lo largo de x). Registra la transformacion para ref."""
    base = dict(x0=240, y0=190, x1=1380, y1=880, xmin=0, xmax=1, ymin=0, ymax=1, xticks=[], yticks=[], yfmt="{:.1f}",
                xlabel="", ylabel="", alfa=1.0, ejes=1.0, series=[], tam_txt=30, rejilla=True)

    def transform(self, p):
        fx = lambda v: p["x0"] + (v - p["xmin"]) / (p["xmax"] - p["xmin"]) * (p["x1"] - p["x0"])  # noqa: E731
        fy = lambda v: p["y1"] - (v - p["ymin"]) / (p["ymax"] - p["ymin"]) * (p["y1"] - p["y0"])  # noqa: E731
        return fx, fy

    def dibujar(self, frame, p, ctx):
        fx, fy = self.transform(p)
        T = p["tam_txt"]

        def ejes(d, X, Y, k):
            c_txt, c_grid = _ci(PALETA["gris"]), _ci(PALETA["linea"])
            f = fuente("r", T * k)
            for v in p["yticks"]:
                if p["rejilla"]:
                    d.line([X(p["x0"]), Y(fy(v)), X(p["x1"]), Y(fy(v))], fill=c_grid, width=int(2 * k))
                d.text((X(p["x0"] - 16), Y(fy(v))), p["yfmt"].format(v), font=f, fill=c_txt, anchor="rm")
            for v in p["xticks"]:
                etq = str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:g}"
                d.text((X(fx(v)), Y(p["y1"] + 16)), etq, font=f, fill=c_txt, anchor="ma")
            d.line([X(p["x0"]), Y(p["y1"]), X(p["x1"]), Y(p["y1"])], fill=_ci(PALETA["tinta"]), width=int(3 * k))
            d.line([X(p["x0"]), Y(p["y0"]), X(p["x0"]), Y(p["y1"])], fill=_ci(PALETA["tinta"]), width=int(3 * k))
            fb = fuente("sb", T * 1.1 * k)
            if p["xlabel"]:
                d.text((X((p["x0"] + p["x1"]) / 2), Y(p["y1"] + 64)), p["xlabel"], font=fb, fill=_ci(PALETA["tinta"]), anchor="ma")
            if p["ylabel"]:
                d.text((X(p["x0"] - 20), Y(p["y0"] - 24)), p["ylabel"], font=fb, fill=_ci(PALETA["tinta"]), anchor="rb")
        bbox = (p["x0"] - 140, p["y0"] - 80, p["x1"] + 40, p["y1"] + 120)
        _lienzo(frame, bbox, p["alfa"] * p["ejes"], ejes)

        for s in p["series"]:
            rv = s.get("revela", 1.0)
            al = s.get("alfa", 1.0) * p["alfa"]
            if rv <= 0.001 or al <= 0.003:
                continue
            xs, ys = s["xs"], s["ys"]
            xcorte = xs[0] + (xs[-1] - xs[0]) * rv
            pts = [(fx(x), fy(y)) for x, y in zip(xs, ys)]
            curva = _catmull(pts) if s.get("suave") else pts
            xc = fx(xcorte)
            vis = []
            for i, q in enumerate(curva):
                if q[0] <= xc + 1e-6:
                    vis.append(q)
                else:
                    if i > 0 and vis:
                        a = curva[i - 1]
                        t = (xc - a[0]) / (q[0] - a[0]) if q[0] != a[0] else 0
                        vis.append((xc, a[1] + (q[1] - a[1]) * t))
                    break
            g = s.get("grosor", 6)
            cc = _ci(color(s.get("color", "tinta")))

            def fn(d, X, Y, k, s=s, vis=vis, g=g, cc=cc, xc=xc, pts=pts):
                if s.get("linea", True) and len(vis) >= 2:
                    d.line([(X(x), Y(y)) for x, y in vis], fill=cc, width=int(g * k), joint="curve")
                mk = s.get("marcadores", 0)
                if mk:
                    for x, y in pts:
                        if x <= xc + 1e-6:
                            d.ellipse([X(x - mk), Y(y - mk), X(x + mk), Y(y + mk)], fill=cc,
                                      outline=_ci(PALETA["blanco"]), width=int(2 * k))
                if 0.001 < rv < 0.999 and vis and s.get("lapiz", True):
                    x, y = vis[-1]
                    rr = g * 1.8
                    d.ellipse([X(x - rr), Y(y - rr), X(x + rr), Y(y + rr)], fill=cc)
            _lienzo(frame, (p["x0"] - 30, p["y0"] - 40, p["x1"] + 30, p["y1"] + 30), al, fn)


class Candado(Elemento):
    base = dict(x=960, y=540, tam=120, color="rojo", abierto=0.0, alfa=1.0, escala=1.0, dy=0.0, disco=True)

    def dibujar(self, frame, p, ctx):
        s = p["tam"] * p["escala"]
        cx, cy = p["x"], p["y"] + p["dy"]

        def fn(d, X, Y, k):
            c = _ci(p["color"])
            if p["disco"]:
                d.ellipse([X(cx - s * 0.78), Y(cy - s * 0.82), X(cx + s * 0.78), Y(cy + s * 0.74)], fill=(255, 255, 255),
                          outline=c, width=int(4 * k))
            bw, bh = s * 0.72, s * 0.52
            by0 = cy - s * 0.05
            lift = s * 0.22 * p["abierto"]
            aw = s * 0.46
            d.arc([X(cx - aw / 2), Y(by0 - s * 0.46 - lift), X(cx + aw / 2), Y(by0 + s * 0.02 - lift)], 180, 360, fill=c,
                  width=int(s * 0.1 * k))
            d.line([X(cx + aw / 2 - s * 0.05), Y(by0 - s * 0.22 - lift), X(cx + aw / 2 - s * 0.05), Y(by0 - lift * 0.2)],
                   fill=c, width=int(s * 0.1 * k))
            d.line([X(cx - aw / 2 + s * 0.05), Y(by0 - s * 0.22 - lift), X(cx - aw / 2 + s * 0.05),
                    Y(by0 - lift - (s * 0.02 if p["abierto"] > 0.5 else -s * 0.05))], fill=c, width=int(s * 0.1 * k))
            d.rounded_rectangle([X(cx - bw / 2), Y(by0), X(cx + bw / 2), Y(by0 + bh)], radius=int(s * 0.08 * k), fill=c)
            d.ellipse([X(cx - s * 0.06), Y(by0 + bh * 0.3), X(cx + s * 0.06), Y(by0 + bh * 0.3 + s * 0.12)], fill=(255, 255, 255))
        _lienzo(frame, (cx - s, cy - s * 1.1, cx + s, cy + s), p["alfa"], fn)


class Marca(Elemento):
    """Visto (forma="check") o cruz (forma="cruz") que se traza con progreso."""
    base = dict(x=960, y=540, tam=60, forma="check", color="verde", progreso=1.0, alfa=1.0, grosor=0)

    def dibujar(self, frame, p, ctx):
        s, cx, cy, pr = p["tam"], p["x"], p["y"], max(0, min(1, p["progreso"]))
        g = p["grosor"] or s * 0.16
        if p["forma"] == "check":
            trazos = [[(cx - s * 0.45, cy), (cx - s * 0.12, cy + s * 0.32), (cx + s * 0.5, cy - s * 0.38)]]
        else:
            trazos = [[(cx - s * 0.4, cy - s * 0.4), (cx + s * 0.4, cy + s * 0.4)],
                      [(cx + s * 0.4, cy - s * 0.4), (cx - s * 0.4, cy + s * 0.4)]]

        def fn(d, X, Y, k):
            c = _ci(p["color"])
            tot = len(trazos)
            for j, tr in enumerate(trazos):
                pj = max(0, min(1, pr * tot - j))
                if pj <= 0:
                    continue
                segs = list(zip(tr[:-1], tr[1:]))
                L = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
                quedan = sum(L) * pj
                pts = [tr[0]]
                for (a, b), l in zip(segs, L):
                    if quedan >= l:
                        pts.append(b)
                        quedan -= l
                    else:
                        f = quedan / l
                        pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                        break
                d.line([(X(x), Y(y)) for x, y in pts], fill=c, width=int(g * k), joint="curve")
        _lienzo(frame, (cx - s, cy - s, cx + s, cy + s), p["alfa"], fn)


class Tarjeta(Elemento):
    """Tarjeta tipo hoja (cuaderno / examen): titulo + items que aparecen (visibles) + marcas por item."""
    base = dict(x=960, y=540, w=520, h=480, titulo="", items=[], visibles=99.0, color="violeta", alfa=1.0,
                escala=1.0, marcas=[], tam=40, dy=0.0)

    def dibujar(self, frame, p, ctx):
        w, h = p["w"] * p["escala"], p["h"] * p["escala"]
        x0, y0 = p["x"] - w / 2, p["y"] + p["dy"] - h / 2
        x1, y1 = x0 + w, y0 + h
        c = _ci(p["color"])
        T = p["tam"] * p["escala"]

        def fn(d, X, Y, k):
            d.rounded_rectangle([X(x0 + 3), Y(y0 + 8), X(x1 + 3), Y(y1 + 8)], radius=int(22 * k), fill=(226, 232, 240))
            d.rounded_rectangle([X(x0), Y(y0), X(x1), Y(y1)], radius=int(22 * k), fill=(255, 255, 255), outline=c,
                                width=int(4 * k))
            d.rounded_rectangle([X(x0), Y(y0), X(x1), Y(y0 + 96 * p["escala"])], radius=int(22 * k), fill=c)
            d.rectangle([X(x0), Y(y0 + 60 * p["escala"]), X(x1), Y(y0 + 96 * p["escala"])], fill=c)
            d.text((X((x0 + x1) / 2), Y(y0 + 48 * p["escala"])), p["titulo"], font=fuente("b", T * 1.05 * k),
                   fill=(255, 255, 255), anchor="mm")
            paso = (h - 130 * p["escala"]) / max(len(p["items"]), 1)
            for i, it in enumerate(p["items"]):
                f = max(0.0, min(1.0, p["visibles"] - i))
                if f <= 0:
                    continue
                yy = y0 + 130 * p["escala"] + paso * (i + 0.5) - paso * 0.5 + 24
                col = tuple(int(255 - (255 - v) * f) for v in _ci(PALETA["tinta"]))
                d.text((X(x0 + 40 * p["escala"]), Y(yy)), it, font=fuente("r", T * k), fill=col, anchor="lm")
                d.line([X(x0 + 40 * p["escala"]), Y(yy + T * 0.85), X(x1 - 40 * p["escala"]), Y(yy + T * 0.85)],
                       fill=tuple(int(255 - (255 - v) * f) for v in (214, 222, 233)), width=int(2 * k))
        _lienzo(frame, (x0 - 10, y0 - 10, x1 + 14, y1 + 18), p["alfa"], fn)
        # marcas por item (check/cruz) trazadas a la derecha
        paso = (h - 130 * p["escala"]) / max(len(p["items"]), 1)
        for i, m in enumerate(p["marcas"]):
            if not m or m.get("progreso", 0) <= 0:
                continue
            yy = y0 + 130 * p["escala"] + paso * i + 24
            Marca().dibujar(frame, dict(Marca.base, x=x1 - 70 * p["escala"], y=yy, tam=T * 1.2, forma=m.get("forma", "check"),
                                        color=color(m.get("color", "verde")), progreso=m["progreso"], alfa=p["alfa"]), ctx)


TIPOS = dict(texto=Texto, numero=Numero, caja=Caja, anillo=Anillo, linea=Linea, puntos=Puntos, grafico=Grafico,
             candado=Candado, marca=Marca, tarjeta=Tarjeta)

ENTRADAS = {
    "fade": dict(alfa=0.0),
    "subir": dict(alfa=0.0, dy=28.0),
    "pop": dict(alfa=0.0, escala=0.55),
    "ninguna": {},
}


def registrar_tipo(nombre, clase):
    """Para tipos propios de una clase/capitulo (el modulo debe listarse en guion["modulos"])."""
    TIPOS[nombre] = clase


# ================================================================ linea de tiempo
class Pista:
    """Un elemento: tipo + segmentos (t0, dur, desde, hasta, ease)."""

    def __init__(self, ident, tipo, t_ini, orden):
        self.id, self.tipo, self.t_ini, self.orden = ident, tipo, t_ini, orden
        self.seg = []         # (t0, dur, desde, hasta, ease)
        self.t_fin = 1e18
        self._t0s = []

    def evaluar(self, t, margen=0.0):
        """Props en t y si hay transicion activa (margen extiende 'activa' para cerrar la ultima frame)."""
        i = bisect.bisect_right(self._t0s, t) - 1
        if i < 0:
            return None, False
        t0, dur, a, b, e = self.seg[i]
        activa = t < t0 + dur + margen
        if dur > 0 and t < t0 + dur:
            return TIPOS[self.tipo]().mezclar(a, b, (t - t0) / dur, e), activa
        return b, activa

    def agregar(self, t0, dur, hasta, e):
        if self.seg:
            actual, _ = self.evaluar(t0)
            desde = actual if actual is not None else self.seg[0][2]
        else:
            desde = hasta
        self.seg.append((t0, dur, desde, hasta, e))
        self.seg.sort(key=lambda s: s[0])
        self._t0s = [s[0] for s in self.seg]


def _props_finales(pista):
    return pista.seg[-1][3] if pista.seg else None


def construir_linea_tiempo(guion, audio_plan):
    """Ensambla audio + cues + pistas animadas. Devuelve dict serializable para los procesos de render."""
    pcm = bytearray(b"\x00\x00" * int(SR * 0.25))
    cues, eventos, beats_t, secciones = [], [], [], []
    avisos = []
    for bi, (beat, frs) in enumerate(zip(guion["beats"], audio_plan)):
        t_beat = len(pcm) / 2 / SR
        marcas_t = {}
        for fr in frs:
            t0 = len(pcm) / 2 / SR
            pal = fr["palabras"]
            for nombre, (idx, esperado) in fr["marcas"].items():
                j = min(idx, len(pal) - 1)
                if esperado and _norm(pal[j][2]) != esperado:
                    cerca = [q for q in range(max(0, j - 3), min(len(pal), j + 4)) if _norm(pal[q][2]) == esperado]
                    if cerca:
                        j = min(cerca, key=lambda q: abs(q - idx))
                    else:
                        avisos.append(f"beat {bi} marca {nombre}: esperaba '{esperado}', TTS dice '{pal[j][2]}'")
                marcas_t[nombre] = t0 + pal[j][0]
            pcm += fr["pcm"]
            cues.append((fr["texto"], t0, len(pcm) / 2 / SR))
            pcm += b"\x00\x00" * int(SR * GAP_FRASE)
        pcm += b"\x00\x00" * int(SR * beat.get("pausa", 0.0))
        t_fin_beat = len(pcm) / 2 / SR
        beats_t.append((t_beat, t_fin_beat))
        if beat.get("seccion"):
            secciones.append((beat["seccion"], t_beat))
        for a in beat.get("acciones", []):
            t = t_beat
            if "en" in a:
                if a["en"] not in marcas_t:
                    raise KeyError(f"beat {bi}: marca [[{a['en']}]] no existe en say")
                t = max(t_beat, marcas_t[a["en"]] - ANTICIPO)
            t += a.get("t", 0.0)
            eventos.append((t, bi, len(eventos), a))
    pcm += b"\x00\x00" * int(SR * COLA)
    total = len(pcm) / 2 / SR
    n_frames = int(math.ceil(total * FPS))
    pcm += b"\x00\x00" * (int(n_frames / FPS * SR) - len(pcm) // 2)
    total = n_frames / FPS

    pistas, todas = {}, []
    orden = 0
    eventos.sort(key=lambda e: (e[0], e[2]))
    for t, bi, _, a in eventos:
        dur = a.get("dur", 0.6)
        e = a.get("ease", "inout")
        if a.get("limpiar") or "quitar" in a:
            ids = a["quitar"] if "quitar" in a else [k for k, p in pistas.items()
                                                     if p.t_fin > t and p.t_ini <= t and k not in a.get("mantener", [])]
            for k in ids:
                p = pistas.get(k)
                if p is None or p.t_fin <= t:
                    continue
                fin = dict(_props_finales(p))
                fin["alfa"] = 0.0
                p.agregar(t, a.get("dur", 0.45), fin, "inout")
                p.t_fin = t + a.get("dur", 0.45)
            continue
        ident = a["id"]
        cambios = {k: v for k, v in a.items() if k not in ("id", "tipo", "en", "t", "dur", "ease", "entrada", "desde")}
        cambios = _resolver_colores(cambios)
        p = pistas.get(ident)
        if p is None or p.t_fin <= t or "tipo" in a:
            if "tipo" not in a:
                raise KeyError(f"beat {bi}: elemento '{ident}' no existe (falta 'tipo')")
            if p is not None and p.t_fin > t:
                viejo = dict(_props_finales(p))
                viejo["alfa"] = 0.0
                p.agregar(t, 0.3, viejo, "inout")
                p.t_fin = t + 0.3
            clase = TIPOS[a["tipo"]]
            props = json.loads(json.dumps(_resolver_colores(dict(clase.base))))
            for k, v in cambios.items():
                if "." in k:
                    _set_ruta(props, k, v)
                else:
                    props[k] = v
            props = clase().preparar(props)
            inicio = dict(props)
            entrada = a.get("entrada", "fade")
            inicio.update(ENTRADAS[entrada])
            inicio.update(_resolver_colores(a.get("desde", {})))
            inicio = clase().preparar(inicio)
            orden += 1
            p = Pista(ident, a["tipo"], t, a.get("z", orden))
            p.seg.append((t, 0.0, inicio, inicio, e))
            p._t0s = [t]
            p.agregar(t, dur if entrada != "ninguna" or a.get("desde") else 0.0, props, a.get("ease", "back" if entrada == "pop" else "out"))
            pistas[ident] = p
            todas.append(p)
            continue
        fin = json.loads(json.dumps(_props_finales(p), default=list))
        fin = _resolver_colores(fin)
        for k, v in cambios.items():
            if "." in k:
                _set_ruta(fin, k, v)
            else:
                fin[k] = v
        fin = TIPOS[p.tipo]().preparar(fin)
        p.agregar(t, dur, fin, e)

    lista = sorted(todas, key=lambda q: q.orden)
    return dict(pcm=bytes(pcm), cues=cues, pistas=lista, total=total, n_frames=n_frames, beats=beats_t,
                secciones=secciones, avisos=avisos, n_acciones=len(eventos))


# ================================================================ render
def _fondo(guion):
    img = Image.new("RGB", (W, H), PALETA["fondo"])
    d = ImageDraw.Draw(img)
    cab = guion.get("cabecera", f"{guion['clase']}  ·  {guion['titulo']}").upper()
    d.rectangle([64, 46, 72, 78], fill=PALETA["teal"])
    d.text((88, 62), cab, font=fuente("b", 26), fill=PALETA["gris"], anchor="lm")
    d.rectangle([0, H - 8, W, H], fill=PALETA["linea"])
    return img


def _activos(pistas, t):
    """Elementos visibles en t y si la escena cambia respecto de la frame anterior."""
    activos, sucio = [], False
    for pista in pistas:
        if t < pista.t_ini or t >= pista.t_fin:
            if pista.t_ini <= t and t - 1.5 / FPS < pista.t_fin:
                sucio = True
            continue
        props, anim = pista.evaluar(t, margen=1.5 / FPS)
        if props is None:
            continue
        el = TIPOS[pista.tipo]()
        if anim or el.es_vivo(props):
            sucio = True
        activos.append((pista, el, props))
    return activos, sucio


def dibujar_escena(activos, t, fondo):
    frame = fondo.copy()
    ctx = {"t": t, "graf": {}}
    for pista, el, props in activos:
        if isinstance(el, Grafico):
            ctx["graf"][pista.id] = el.transform(props)
    for pista, el, props in sorted(activos, key=lambda q: q[0].orden):
        if "ref" in props and props["ref"] in ctx["graf"]:
            fx, fy = ctx["graf"][props["ref"]]
            props = dict(props)
            for k in ("x", "x0", "x1"):
                if k in props:
                    props[k] = fx(props[k])
            for k in ("y", "y0", "y1"):
                if k in props:
                    props[k] = fy(props[k])
            if "x0" in props and pista.tipo == "caja":
                props = Caja().preparar(props)
        try:
            el.dibujar(frame, props, ctx)
        except Exception as e:
            raise RuntimeError(f"elemento '{pista.id}' ({pista.tipo}) en t={t:.2f}: {e}") from e
    return frame


def _barra(img, t, total):
    ImageDraw.Draw(img).rectangle([0, H - 8, int(W * t / total), H], fill=PALETA["teal"])


def _cargar(datos_pkl):
    global FPS
    with open(datos_pkl, "rb") as fh:
        D = pickle.load(fh)
    FPS = D["fps"]
    for mod in D["modulos"]:
        __import__(mod)
    return D


def _render_trozo(args):
    datos_pkl, f_ini, f_fin, salida = args
    D = _cargar(datos_pkl)
    fondo = _fondo(D["guion"])
    pistas, total = D["pistas"], D["total"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-tune", "animation", "-pix_fmt", "yuv420p",
           "-g", str(FPS * 2), "-threads", "2", str(salida)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    escena, ultimo_ancho, buf = None, -1, None
    for fi in range(f_ini, f_fin):
        t = fi / FPS
        activos, sucio = _activos(pistas, t)
        if sucio or escena is None:
            escena = dibujar_escena(activos, t, fondo)
            ultimo_ancho = -1
        ancho = int(W * t / total)
        if ancho != ultimo_ancho:
            fr = escena.copy()
            _barra(fr, t, total)
            buf = fr.tobytes()
            ultimo_ancho = ancho
        proc.stdin.write(buf)
    proc.stdin.close()
    proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg fallo en trozo {salida}")
    return str(salida)


def previsualizar(tmp, tiempos, carpeta):
    """Renderiza frames sueltos (PNG) desde la linea de tiempo ya construida en tmp."""
    D = _cargar(Path(tmp) / "linea_tiempo.pkl")
    fondo = _fondo(D["guion"])
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    salidas = []
    for t in tiempos:
        activos, _ = _activos(D["pistas"], t)
        img = dibujar_escena(activos, t, fondo)
        _barra(img, t, D["total"])
        f = carpeta / f"frame_{t:07.2f}.png"
        img.save(f)
        salidas.append(f)
    return salidas


def construir(guion_json, salida_mp4, tmp, procesos=None, solo_audio=False):
    guion_json, salida_mp4, tmp = Path(guion_json), Path(salida_mp4), Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    guion = json.loads(guion_json.read_text(encoding="utf-8"))
    global FPS
    FPS = guion.get("fps", FPS)
    plan = sintetizar(guion, tmp / "tts")
    L = construir_linea_tiempo(guion, plan)
    for a in L["avisos"]:
        print("AVISO", a)

    wav = tmp / "audio.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(L["pcm"])
    srt = salida_mp4.with_suffix(".srt")
    vc.escribir_srt(L["cues"], srt)
    informe = dict(duracion=L["total"], beats=len(guion["beats"]), acciones=L["n_acciones"],
                   beats_t=L["beats"], secciones=L["secciones"])
    (tmp / "informe.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    datos = dict(guion=guion, pistas=L["pistas"], total=L["total"], modulos=guion.get("modulos", []), fps=FPS)
    pkl = tmp / "linea_tiempo.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(datos, fh)
    if solo_audio:
        return L, informe
    procesos = procesos or max(2, (os.cpu_count() or 4) // 2)
    n = L["n_frames"]
    cortes = [round(n * i / procesos) for i in range(procesos + 1)]
    trozos = [(str(pkl), cortes[i], cortes[i + 1], tmp / f"trozo_{i:02}.mp4") for i in range(procesos)
              if cortes[i + 1] > cortes[i]]
    print(f"Renderizando {n} frames ({L['total']:.1f} s) en {len(trozos)} procesos...")
    with ProcessPoolExecutor(max_workers=len(trozos)) as ex:
        list(ex.map(_render_trozo, trozos))
    (tmp / "trozos.txt").write_text("\n".join(f"file '{Path(t[3]).as_posix()}'" for t in trozos), encoding="utf-8")

    meta = [";FFMETADATA1", f"title={guion['clase']} · {guion['titulo']}", "artist=" + guion.get("profesora", "")]
    secs = L["secciones"] or [(guion["titulo"], 0.0)]
    for k, (nombre, ini) in enumerate(secs):
        fin = secs[k + 1][1] if k + 1 < len(secs) else L["total"]
        ini = 0.0 if k == 0 else ini
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(ini * 1000)}", f"END={int(fin * 1000)}", f"title={nombre}"]
    (tmp / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    # 1) une los trozos de video (copia) con timescale 1000. La pista de capitulos del MP4 usa el MCM de los
    #    timescales (mov_text = 1/1e6, audio 1/48000, video): con video a 1/1000 queda en ~1/6e6 y cada
    #    capitulo debe durar < 2^31 ticks (~350 s); con 1/12800 subia a 1/24e6 y desbordaba con ~90 s.
    secs_t = [s[1] for s in L["secciones"]] + [L["total"]]
    if any(b - a > 340 for a, b in zip(secs_t, secs_t[1:])):
        print("AVISO: una seccion dura > 340 s; su capitulo MP4 puede desbordar (marca mas 'seccion')")
    video = tmp / "video_sin_audio.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "trozos.txt"),
                    "-c", "copy", "-video_track_timescale", "1000", str(video)], check=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-i", str(wav), "-i", str(srt), "-i", str(tmp / "meta.txt"),
           "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3", "-map_chapters", "3",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-c:s", "mov_text",
           "-metadata:s:s:0", "language=spa", "-metadata:s:a:0", "language=spa", "-t", f"{L['total']:.3f}",
           "-movflags", "+faststart", str(salida_mp4)]
    subprocess.run(cmd, check=True)
    print(f"OK {salida_mp4}  ({L['total'] / 60:.2f} min, {len(guion['beats'])} beats, {L['n_acciones']} acciones)")
    return L, informe
