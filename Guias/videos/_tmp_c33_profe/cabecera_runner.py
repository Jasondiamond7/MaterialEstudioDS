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

