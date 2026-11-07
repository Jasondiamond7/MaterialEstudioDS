"""Verificacion de la serie C35: duraciones (ffprobe), capitulos y silencios > 1.5 s a -40 dB (silencedetect)."""
import json
import re
import subprocess
from pathlib import Path

C35 = Path(__file__).resolve().parents[1] / "C35"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout.strip())


def streams(p):
    js = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(p)], capture_output=True, text=True,
                        encoding="utf-8", errors="ignore").stdout
    return [(s["codec_type"], s.get("codec_name"), s.get("width"), s.get("height")) for s in json.loads(js)["streams"]]


def caps(p):
    js = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(p)], capture_output=True, text=True,
                        encoding="utf-8", errors="ignore").stdout
    return [(float(c["start_time"]), float(c["end_time"]), c["tags"].get("title", "")) for c in json.loads(js)["chapters"]]


def silencios(p, umbral=1.5):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(p), "-af", f"silencedetect=noise=-40dB:d={umbral}", "-vn", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="ignore").stderr
    return [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", err)]


tot = 0.0
for f in sorted(C35.glob("C35_[0-9][0-9]_*.mp4")):
    d = dur(f)
    tot += d
    cs = caps(f)
    print(f"{f.name:34s} {d:7.1f} s  caps={len(cs):2d} max_cap={max(b - a for a, b, _ in cs):5.1f}  silencios>1.5s={silencios(f)}  {streams(f)}")
full = C35 / "C35_Completo_RandomForest.mp4"
if full.exists():
    cs = caps(full)
    print(f"\n{full.name}: {dur(full):.1f} s (suma partes {tot:.1f} s, dif {dur(full) - tot:+.2f} s)  capitulos={len(cs)}  "
          f"max_cap={max(b - a for a, b, _ in cs):.1f} s  {streams(full)}")
    for a, b, t in cs:
        print(f"  {a:7.1f} {b:7.1f} {b - a:6.1f}  {t}")
    print("silencios>1.5s:", silencios(full))
