"""Parche tras la vuelta de Whisper (ronda 4): narraciones que fallaban y equivalencias de escritura de Whisper."""
from pathlib import Path

AQUI = Path(__file__).resolve().parent
G = AQUI / "generadores"
edits = {
    "gen_09.py": [("si no mejora durante ka rondas seguidas, la paciencia,", "si no mejora durante varias rondas seguidas, la paciencia,"),
                  ("y para cuando deja de mejorar durante ka rondas.", "y para cuando deja de mejorar durante varias rondas seguidas."),
                  ("salvo que actives n_iter_no_change. [[b]]Por defecto ese parámetro está apagado,",
                   "salvo que actives su propia parada temprana. [[b]]Por defecto ese parámetro está apagado,")],
    "gen_11.py": [("y para tras ka rondas sin mejorar.", "y para tras varias rondas seguidas sin mejorar."),
                  ("Las implementaciones: [[a]]XGBoost agrega un objetivo regularizado,",
                   "Las implementaciones: [[a]]primero, XGBoost, que agrega un objetivo regularizado,")],
    "gen_04.py": [("Ningún stump razonable lo clasifica bien,", "Ningún árbol razonable lo clasifica bien,")],
}
for f, rs in edits.items():
    q = G / f
    t = q.read_text(encoding="utf-8")
    for a, b in rs:
        if b in t:
            continue
        assert a in t, (f, a)
        t = t.replace(a, b)
    q.write_text(t, encoding="utf-8")

w = AQUI / "whisper_check.py"
t = w.read_text(encoding="utf-8")
a = '           (["cat", "boost"], ["catbust"])]'
if a in t:
    t = t.replace(a, '           (["cev"], ["ce", "uve"]), (["psykit"], ["saikit"]), (["psyche"], ["saikit"]),\n'
                     '           (["gradientboosting"], ["greidien", "boosting"]), (["erly", "stopping"], ["er", "li", "stoping"]),\n'
                     '           (["xgbost"], ["ecs", "yi", "bust"]), (["en", "stimators"], ["en", "estimeitors"])]')
    w.write_text(t, encoding="utf-8")
print("ok")
