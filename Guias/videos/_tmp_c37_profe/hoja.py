"""Hoja de contacto: python hoja.py N t1 t2 ... -> _tmp_c37_profe/NN/previa/hoja.png (frames a 640x360 en rejilla de 3)."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "_fuente"))
import video_c37_completo as R  # noqa: E402
import video_profe as vp  # noqa: E402

n = int(sys.argv[1])
tmp = AQUI / f"{n:02}"
if sys.argv[2] == "--beats":            # fin de cada beat (composicion completa), en hojas de 12
    import json
    bt = json.loads((tmp / "informe.json").read_text(encoding="utf-8"))["beats_t"]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ts = [round(b - 0.35, 2) for _a, b in bt][12 * k:12 * k + 12]
else:
    ts = [float(x) for x in sys.argv[2:]]
fr = vp.previsualizar(tmp, ts, tmp / "previa")
w, h = 640, 360
cols = 3
filas = (len(fr) + cols - 1) // cols
hoja = Image.new("RGB", (w * cols, h * filas), "white")
for i, (f, t) in enumerate(zip(fr, ts)):
    im = Image.open(f).resize((w, h), Image.LANCZOS)
    ImageDraw.Draw(im).text((8, 8), f"{t:.1f}s", fill=(220, 0, 0))
    hoja.paste(im, ((i % cols) * w, (i // cols) * h))
salida = tmp / "previa" / f"hoja_{int(ts[0])}.png"
hoja.save(salida)
print(salida)
