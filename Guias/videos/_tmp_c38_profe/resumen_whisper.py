"""Resumen de la vuelta de Whisper: WER por seccion y terminos no recuperados. python resumen_whisper.py [c38|base]"""
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
W = Path(__file__).resolve().parent / "whisper"
modo = sys.argv[1] if len(sys.argv) > 1 else "c38"
te = tr = 0
for f in sorted(W.glob(f"res_{modo}_*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    te += r["errores"]; tr += r["palabras_ref"]
    print(f"{f.stem}: WER {100 * r['wer']:.2f} % ({r['errores']}/{r['palabras_ref']})")
    for t, e in sorted(r["terminos"].items()):
        u = e["n"] - e["no_ubicado"]
        if e["recuperado"] < u:
            print(f"   MAL {t} {e['recuperado']}/{u} voz='{e['voz']}' oido={e['oido'][:3]}")
    print("   top", [(w, c, o) for w, c, o in r["no_recuperadas"][:12]])
print(f"TOTAL {modo}: WER {100 * te / max(tr, 1):.2f} % ({te}/{tr})")
