"""Genera Guias/videos/C30_Video_PreparacionML.mp4 (+ .srt) desde guion_c30.json.

Ejecutar con el venv de Guias/videos/.venv:
    Guias/videos/.venv/Scripts/python.exe Guias/_fuente/video_c30.py
"""
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import video_comun as vc  # noqa: E402

RAIZ = AQUI.parent.parent
TMP = RAIZ / "Guias" / "videos" / "_tmp_c30"
SALIDA = RAIZ / "Guias" / "videos" / "C30_Video_PreparacionML.mp4"

if __name__ == "__main__":
    informe, caps, guion = vc.construir_video(AQUI / "guion_c30.json", RAIZ, SALIDA, TMP)
    for ci, ini in caps:
        m, s = divmod(int(ini), 60)
        print(f"{m:02}:{s:02}  {ci + 1}. {guion['capitulos'][ci]['titulo']}")
