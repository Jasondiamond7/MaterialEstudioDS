"""Vuelca el texto EXACTO enviado al TTS (post-sustitucion) por frase, para la vuelta de Whisper.
python dump_voz.py N [base]  ->  whisper/voz_{c36|base}_NN.json  (orden = orden del audio)."""
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "_fuente"))
import video_c36_completo as R  # noqa: E402
import video_profe as vp  # noqa: E402

TERMS = ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "boosting", "Boosting", "bagging", "Random Forest",
         "learning_rate", "n_estimators", "max_depth", "subsample", "subsample_freq", "num_leaves", "n_iter_no_change",
         "early stopping", "Early stopping", "weak learner", "shrinkage", "loss", "log-loss", "stump", "stumps", "gradient",
         "leaf-wise", "level-wise", "GOSS", "EFB", "Huber", "adaptive", "tuning", "bins", "one-hot", "staged_predict",
         "GridSearchCV", "RandomizedSearchCV", "permutation_importance", "feature_importances_", "scikit-learn", "Python",
         "Pipeline", "notebook", "Heart", "Ames", "AUC", "RMSE", "churn", "η", "ε", "α", "≈", "%", "×", "·"]
n = int(sys.argv[1])
modo = sys.argv[2] if len(sys.argv) > 2 else "c36"
if modo == "base":
    R.modo_base()
g = json.loads((R.AQUI / R._por_numero(n)[2]).read_text(encoding="utf-8"))
out, dec_miles = [], 0
for b in g["beats"]:
    for f in vp.frases(b["say"]):
        pant = vp.a_pantalla(f)
        voz = vp.a_voz(f)
        dec_miles += len(re.findall(r"(?<![\w.,])[1-9]\d{0,2}\.\d{3}(?![\d\w])", pant))
        ts = []
        for t in TERMS:
            pat = r"(?<![\w])" + re.escape(t) + r"(?![\w])" if t[0].isalnum() or t[0] == "'" else re.escape(t)
            k = len(re.findall(pat, pant))
            if k:
                ts.append([t, vp.a_voz(t), k])
        out.append(dict(pantalla=pant, voz=voz, terms=ts))
# equivalencias termino original -> respelling, SOLO de la tabla del modo activo (para que "XGBoost" escrito por Whisper
# cuente como recuperado cuando la voz dijo "ecs yi búst")
eq = []
tabla = R.R35.PRON_C35 if modo == "base" else R.PRON_C36
for orig, rep in tabla + list(vp.vc.PRONUNCIACION):
    o = orig.replace("\\b", "").replace("?", "").replace("'", "").strip("_")
    if re.fullmatch(r"[A-Za-z0-9 _\-]+", o) and len(o) > 1:
        eq.append([o.replace("_", " ").replace("-", " "), rep])
        cc = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", o).replace("_", " ").replace("-", " ")    # LightGBM -> Light GBM
        if cc != o.replace("_", " ").replace("-", " "):
            eq.append([cc, rep])
(AQUI / "whisper").mkdir(exist_ok=True)
(AQUI / "whisper" / f"voz_{modo}_{n:02}.json").write_text(json.dumps(dict(frases=out, decimales_leidos_como_miles=dec_miles, equivalencias=eq),
                                                                    ensure_ascii=False, indent=0), encoding="utf-8")
print(modo, n, len(out), "frases; decimales N.NNN (N>=1) en pantalla:", dec_miles)
