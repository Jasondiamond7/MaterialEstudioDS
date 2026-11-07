"""Verificacion de la serie C34: duraciones, capitulos y silencios > 1.5 s."""
import json
import re
import subprocess
from pathlib import Path

C34 = Path(__file__).resolve().parents[1] / "C34"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout.strip())


def caps(p):
    js = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(p)], capture_output=True, text=True).stdout
    return [(float(c["start_time"]), float(c["end_time"]), c["tags"].get("title", "")) for c in json.loads(js)["chapters"]]


def silencios(p, umbral=1.5):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(p), "-af", f"silencedetect=noise=-40dB:d={umbral}", "-vn", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", err)]


tot = 0.0
for f in sorted(C34.glob("C34_[0-9][0-9]_*.mp4")):
    d = dur(f)
    tot += d
    print(f"{f.name:40s} {d:7.1f} s  caps={len(caps(f)):2d}  silencios>1.5s={silencios(f)}")
full = C34 / "C34_Completo_ArbolesDecision.mp4"
if full.exists():
    cs = caps(full)
    print(f"\n{full.name}: {dur(full):.1f} s (suma partes {tot:.1f} s)  capitulos={len(cs)}  max_cap={max(b - a for a, b, _ in cs):.1f} s")
    for a, b, t in cs:
        print(f"  {a:7.1f} {b:7.1f} {b - a:6.1f}  {t}")
    print("silencios>1.5s:", silencios(full))
