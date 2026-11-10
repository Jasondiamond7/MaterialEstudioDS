"""Extrae 3 frames por MP4 final (C39/) al final de beats con numeros ya contados (fin de beat - 0.4 s; los conteos duran 1 s
y arrancan antes) -> frames_mp4/. Elige, en cada video, tres beats repartidos entre los que tienen elementos 'numero' o graficos."""
import json
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
C39 = AQUI.parent / "C39"
FUENTE = AQUI.parents[1] / "_fuente"
SAL = AQUI / "frames_mp4"
SAL.mkdir(exist_ok=True)
for n in range(1, 13):
    mp4 = next(C39.glob(f"C39_{n:02}_*.mp4"))
    g = json.loads(next(FUENTE.glob(f"guion_c39_{n:02}_*.json")).read_text(encoding="utf-8"))
    cand = [i for i, b in enumerate(g["beats"]) if any(a.get("tipo") in ("numero", "grafico") for a in b["acciones"])]
    cand = cand or list(range(1, len(g["beats"])))
    elegidos = sorted({cand[0], cand[len(cand) // 2], cand[-1]})
    bt = json.loads((AQUI / f"{n:02}" / "informe.json").read_text(encoding="utf-8"))["beats_t"]
    for b in elegidos:
        t = round(bt[b][1] - 0.4, 2)
        out = SAL / f"C39_{n:02}_b{b:02}_t{t:07.2f}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-vf", "scale=960:-1", str(out)],
                       check=True)
        print(out.name)
