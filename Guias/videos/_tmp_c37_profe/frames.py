"""Extrae 3 frames por MP4 final (C37/) al final de beats con numeros ya contados (fin de beat - 0.4 s) -> frames_mp4/."""
import json
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
C37 = AQUI.parent / "C37"
SAL = AQUI / "frames_mp4"
SAL.mkdir(exist_ok=True)
# beats elegidos por video (indice de beat con datos reales o ejemplo resuelto)
ELEGIDOS = {1: [5, 11, 13], 2: [7, 13, 14], 3: [7, 8, 14], 4: [4, 13, 17], 5: [6, 12, 14], 6: [5, 12, 16], 7: [8, 11, 14], 8: [3, 5, 8],
            9: [7, 11, 14], 10: [3, 5, 13], 11: [7, 12, 14], 12: [5, 8, 13]}
for n, bs in ELEGIDOS.items():
    mp4 = next(C37.glob(f"C37_{n:02}_*.mp4"))
    bt = json.loads((AQUI / f"{n:02}" / "informe.json").read_text(encoding="utf-8"))["beats_t"]
    for b in bs:
        t = round(bt[b][1] - 0.4, 2)
        out = SAL / f"C37_{n:02}_t{t:07.2f}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-vf", "scale=960:-1", str(out)],
                       check=True)
        print(out.name)
