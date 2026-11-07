"""Uso: hoja.py N t1 t2 ... -> genera previas del video N y hojas de contacto 2x2 en el scratchpad."""
import sys, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
n = int(sys.argv[1])
if sys.argv[2] == "auto":
    inf = json.load(open(f"C:/Users/Jason Leon/Desktop/MaterialEstudioDS/Guias/videos/_tmp_c31_profe/{n:02}/informe.json", encoding="utf-8"))
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ts = [round(a + min(3.2, (b - a) * 0.8), 1) for i, (a, b) in enumerate(inf["beats_t"]) if i % step == 1 or i == 0]
else:
    ts = [float(x) for x in sys.argv[2:]]
F = Path(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")
py = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\.venv\Scripts\python.exe"
out = subprocess.run([py, "-X", "utf8", str(F / "video_c31_completo.py"), "--previa", str(n)] + [str(t) for t in ts], capture_output=True, text=True, cwd=str(F))
files = [l.strip() for l in out.stdout.splitlines() if l.strip().lower().endswith(".png")]
if not files: print(out.stdout[-800:], out.stderr[-800:]); sys.exit(1)
dest = Path(r"C:\Users\JASONL~1\AppData\Local\Temp\claude\c--Users-Jason-Leon-Desktop-MaterialEstudioDS\fe9714ec-0a13-4f75-8374-2d7c3811070c\scratchpad\c31_profe")
dest.mkdir(parents=True, exist_ok=True)
fnt = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 40)
for h in range(0, len(files), 4):
    sheet = Image.new("RGB", (1920, 1080), "white")
    for j, f in enumerate(files[h:h+4]):
        im = Image.open(f).convert("RGB").resize((960, 540))
        d = ImageDraw.Draw(im); d.text((14, 500), f"t={ts[h+j]:g}", font=fnt, fill=(200, 0, 0))
        sheet.paste(im, ((j % 2) * 960, (j // 2) * 540))
    p = dest / f"v{n:02}_hoja{h//4+1}.png"; sheet.save(p); print(p)
