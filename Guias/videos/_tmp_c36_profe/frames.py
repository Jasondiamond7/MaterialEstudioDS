"""Extrae 3 frames por MP4 final (C36/) al final de beats con numeros ya contados (fin de beat - 0.4 s) -> frames_mp4/."""
import json
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
C36 = AQUI.parent / "C36"
SAL = AQUI / "frames_mp4"
SAL.mkdir(exist_ok=True)
# beats elegidos por video (indice de beat con datos reales o ejemplo resuelto)
ELEGIDOS = {1: [7, 10, 12], 2: [12, 13, 16], 3: [9, 12, 15], 4: [11, 13, 15], 5: [8, 10, 11], 6: [7, 11, 13],
            7: [8, 11, 13], 8: [6, 15, 17], 9: [5, 9, 15], 10: [3, 5, 7], 11: [4, 11, 15]}
for n, bs in ELEGIDOS.items():
    mp4 = next(C36.glob(f"C36_{n:02}_*.mp4"))
    bt = json.loads((AQUI / f"{n:02}" / "informe.json").read_text(encoding="utf-8"))["beats_t"]
    for b in bs:
        t = round(bt[b][1] - 0.4, 2)
        out = SAL / f"C36_{n:02}_t{t:07.2f}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-vf", "scale=960:-1", str(out)],
                       check=True)
        print(out.name)
