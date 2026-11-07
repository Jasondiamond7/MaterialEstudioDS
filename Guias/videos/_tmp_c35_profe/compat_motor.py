"""Compatibilidad: a_voz del motor modificado (opciones apagadas) == motor original, en todas las frases de todos los guiones."""
import importlib.util, json, sys
from pathlib import Path
F = Path(__file__).resolve().parents[2] / "_fuente"
sys.path.insert(0, str(F))
import video_profe as nuevo
spec = importlib.util.spec_from_file_location("orig", Path(__file__).parent / "video_profe_orig.py")
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
# misma lista extra para ambos (los runners la extienden igual)
assert nuevo.NUM_DECIMAL is False and nuevo.SIMBOLOS_VOZ is False
n = dif = 0
for g in sorted(F.glob("guion_c3*.json")):
    if g.name.startswith("guion_c35"):
        continue
    js = json.loads(g.read_text(encoding="utf-8"))
    for b in js.get("beats", []):
        for f in nuevo.frases(b.get("say", "")):
            n += 1
            if nuevo.a_voz(f) != orig.a_voz(f) or nuevo.a_pantalla(f) != orig.a_pantalla(f):
                dif += 1; print("DIF", g.name, f[:80])
print(f"frases comparadas: {n}  diferencias: {dif}")
