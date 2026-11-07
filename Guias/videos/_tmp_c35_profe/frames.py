"""Extrae 3 frames por MP4 final (C35/) en momentos con numeros ya contados (fin de beat - 0.4 s) -> frames_mp4/."""
import json
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
C35 = AQUI.parent / "C35"
SAL = AQUI / "frames_mp4"
SAL.mkdir(exist_ok=True)
# beats elegidos por video (indice de beat con datos o ejemplo resuelto)
ELEGIDOS = {1: [9, 11, 18], 2: [7, 10, 12], 3: [6, 9, 11], 4: [6, 9, 14], 5: [10, 12, 13], 6: [5, 9, 13],
            7: [4, 6, 10], 8: [3, 9, 11], 9: [6, 8, 12], 10: [3, 5, 9], 11: [2, 9, 11]}
for n, bs in ELEGIDOS.items():
    mp4 = next(C35.glob(f"C35_{n:02}_*.mp4"))
    bt = json.loads((AQUI / f"{n:02}" / "informe.json").read_text(encoding="utf-8"))["beats_t"]
    for b in bs:
        t = round(bt[b][1] - 0.4, 2)
        out = SAL / f"C35_{n:02}_t{t:07.2f}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-vf", "scale=960:-1", str(out)],
                       check=True)
        print(out.name)
