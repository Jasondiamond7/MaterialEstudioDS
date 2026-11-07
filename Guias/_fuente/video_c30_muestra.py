"""Capitulo de muestra (formato "profesor experto"): Clase 30 · Entrenamiento, validacion y prueba.

Genera Guias/videos/C30_Muestra_EntrenamientoValidacionPrueba.mp4 (+ .srt) desde guion_c30_muestra.json.

Ejecutar con el venv de Guias/videos/.venv:
    Guias/videos/.venv/Scripts/python.exe Guias/_fuente/video_c30_muestra.py              # video completo
    Guias/videos/.venv/Scripts/python.exe Guias/_fuente/video_c30_muestra.py --previa 12.5 40 95   # PNGs sueltos
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import video_profe as vp  # noqa: E402

RAIZ = AQUI.parent.parent
TMP = RAIZ / "Guias" / "videos" / "_tmp_c30_muestra"
SALIDA = RAIZ / "Guias" / "videos" / "C30_Muestra_EntrenamientoValidacionPrueba.mp4"
GUION = AQUI / "guion_c30_muestra.json"

if __name__ == "__main__":
    if "--previa" in sys.argv:
        tiempos = [float(x) for x in sys.argv[sys.argv.index("--previa") + 1:]]
        vp.construir(GUION, SALIDA, TMP, solo_audio=True)
        for f in vp.previsualizar(TMP, tiempos, TMP / "previa"):
            print(f)
        sys.exit(0)
    L, inf = vp.construir(GUION, SALIDA, TMP)
    minutos = inf["duracion"] / 60
    print(f"beats/min: {inf['beats'] / minutos:.1f}   acciones visuales/min: {inf['acciones'] / minutos:.1f}")
    for nombre, t in inf["secciones"]:
        m, s = divmod(int(t), 60)
        print(f"{m:02}:{s:02}  {nombre}")
    (TMP / "informe.json").write_text(json.dumps(inf, indent=1, ensure_ascii=False), encoding="utf-8")
