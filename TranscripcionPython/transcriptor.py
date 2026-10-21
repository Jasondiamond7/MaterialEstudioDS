"""Transcribe un video (o audio) de una clase a texto usando OpenAI Whisper.

Uso basico:
    python transcriptor.py "C:/Users/Jason Leon/Desktop/D/VideosClases/clase35.mp4"

Opciones utiles:
    python transcriptor.py video.mp4 --model small --formats txt srt
    python transcriptor.py video.mp4 --output-dir salidas --language es
    python transcriptor.py video.mp4 --clip-end 2:05:00   (grabacion que sigue tras la clase)

Genera, por defecto, un .txt (texto corrido) y un .srt (con marcas de tiempo)
junto al video, con el mismo nombre del archivo de entrada.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import whisper
from whisper.audio import SAMPLE_RATE
from whisper.utils import get_writer

# Modelos disponibles, de mas rapido/menos preciso a mas lento/mas preciso.
MODELOS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
FORMATOS = ["txt", "srt", "vtt", "tsv", "json"]


def parse_timecode(valor: str) -> float:
    """Convierte '90', '1:30' o '1:23:45' a segundos (float)."""
    partes = valor.strip().split(":")
    try:
        numeros = [float(p) for p in partes]
    except ValueError:
        raise argparse.ArgumentTypeError(f"tiempo invalido: {valor!r}")
    segundos = 0.0
    for n in numeros:
        segundos = segundos * 60 + n
    return segundos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe un video/audio a texto con Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Ruta del video o audio a transcribir.")
    parser.add_argument(
        "--model", default="medium", choices=MODELOS,
        help="Modelo Whisper. Por defecto 'medium' (calidad alta). "
             "'small' es mas rapido en CPU; 'large-v3' es aun mas preciso pero mas lento.",
    )
    parser.add_argument(
        "--language", default="es",
        help="Idioma del audio (codigo ISO, p. ej. 'es'). Usa 'auto' para detectarlo.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Carpeta de salida. Por defecto, la misma carpeta del video.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["txt", "srt"], choices=FORMATOS,
        help="Formatos de salida a generar.",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Dispositivo de computo. Por defecto se detecta automaticamente.",
    )
    parser.add_argument(
        "--initial-prompt", default=None,
        help="Texto de contexto para mejorar terminos especificos "
             "(p. ej. nombres propios o vocabulario tecnico de la clase).",
    )
    parser.add_argument(
        "--clip-start", type=parse_timecode, default=0.0, metavar="TIEMPO",
        help="Transcribir desde este punto. Segundos o HH:MM:SS / MM:SS. Por defecto, 0.",
    )
    parser.add_argument(
        "--clip-end", type=parse_timecode, default=None, metavar="TIEMPO",
        help="Transcribir hasta este punto (util cuando la grabacion sigue tras la clase). "
             "Segundos o HH:MM:SS / MM:SS. Las marcas de tiempo del .srt quedan relativas a --clip-start.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="No mostrar el texto en pantalla mientras transcribe.",
    )
    return parser.parse_args(argv)


def detectar_device(preferido: str | None) -> str:
    if preferido:
        return preferido
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def formato_tiempo(segundos: float) -> str:
    m, s = divmod(int(segundos), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Validaciones tempranas: fallar rapido con un mensaje claro.
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: no se encontro el ejecutable 'ffmpeg' en el PATH.\n"
            "Whisper lo necesita para leer el audio del video.\n"
            "Instalalo con:  winget install --id Gyan.FFmpeg -e\n"
            "y reinicia la terminal.",
            file=sys.stderr,
        )
        return 1

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"ERROR: no existe el archivo: {video}", file=sys.stderr)
        return 1

    output_dir = (args.output_dir or video.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detectar_device(args.device)
    language = None if args.language.lower() == "auto" else args.language
    fp16 = device == "cuda"  # fp16 en CPU solo genera warnings y no acelera.

    recorte = args.clip_start > 0 or args.clip_end is not None

    print(f"Archivo    : {video}")
    print(f"Modelo     : {args.model}")
    print(f"Dispositivo: {device}")
    print(f"Idioma     : {language or 'auto-deteccion'}")
    print(f"Salida     : {output_dir}")
    if recorte:
        fin = formato_tiempo(args.clip_end) if args.clip_end is not None else "final"
        print(f"Recorte    : {formato_tiempo(args.clip_start)} -> {fin}")
    if device == "cpu" and args.model in {"medium", "large-v3"}:
        print(
            "AVISO: el modelo "
            f"'{args.model}' en CPU puede tardar bastante (varias veces la "
            "duracion del video). Considera --model small si tienes prisa."
        )

    # 2. Carga del modelo (la primera vez descarga los pesos y los cachea).
    print("\nCargando modelo...")
    t0 = time.perf_counter()
    model = whisper.load_model(args.model, device=device)

    # 3. Audio: se pasa la ruta salvo que se pida un recorte, en cuyo caso se
    #    carga el audio completo y se corta el tramo pedido antes de transcribir.
    audio = str(video)
    if recorte:
        muestras = whisper.load_audio(str(video))
        ini = int(args.clip_start * SAMPLE_RATE)
        fin = int(args.clip_end * SAMPLE_RATE) if args.clip_end is not None else len(muestras)
        audio = muestras[ini:fin]
        print(f"Audio recortado: {len(audio) / SAMPLE_RATE / 60:.1f} min de "
              f"{len(muestras) / SAMPLE_RATE / 60:.1f} min totales")

    # 4. Transcripcion.
    print("Transcribiendo... (esto puede tardar)\n")
    result = model.transcribe(
        audio,
        language=language,
        fp16=fp16,
        verbose=None if args.quiet else True,
        initial_prompt=args.initial_prompt,
    )

    elapsed = time.perf_counter() - t0
    idioma_detectado = result.get("language", language or "?")

    # 5. Escritura de resultados en todos los formatos pedidos.
    escritos: list[Path] = []
    for fmt in args.formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(video), {"max_line_width": None, "max_line_count": None})
        escritos.append(output_dir / f"{video.stem}.{fmt}")

    print("\n" + "=" * 60)
    print(f"Listo en {formato_tiempo(elapsed)} (idioma: {idioma_detectado})")
    for ruta in escritos:
        print(f"  -> {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
