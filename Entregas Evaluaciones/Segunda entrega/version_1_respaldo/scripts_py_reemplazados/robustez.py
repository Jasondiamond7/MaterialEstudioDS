"""Robustness checks for the preliminary validation (Entrega 2).

Answers four questions a reviewer would ask about the preliminary results:
1. Does the model beat the baseline every year, or only on average?
2. Is the baseline fair? The real practice uses every past harvest, including
   2019 and the cuarteles without canopy signal.
3. Is the result inflated by choosing the canopy variant and the model with the
   same years used to report the error (selection bias)? Re-select using only
   2022-2025 and evaluate the choice once on 2026, which was never used to choose.
4. Is the improvement over the baseline larger than sampling noise? Paired
   bootstrap, resampling whole cuarteles (cluster bootstrap).

Run validacion_preliminar.py first: this script reads its prediction files.
"""
import json

import numpy as np
import pandas as pd

from construir_panel import CARPETA_SALIDA, construir_panel

VARIANTES = ['cnp_acum_dic', 'cnp_acum_ene', 'cnp_acum_feb', 'cnp_media_ene', 'cnp_media_completa']
MODELOS = ['M1 Ridge (canopia + cuartel)', 'M2 Modelo mixto (cuartel aleatorio)',
           'M3 Random Forest', 'M4 Gradient Boosting']
BASELINE = 'B0 Promedio historico del cuartel'
PRINCIPAL = ('cnp_acum_ene', 'M2 Modelo mixto (cuartel aleatorio)')
ANIOS_SELECCION = [2022, 2023, 2024, 2025]
ANIO_CIEGO = 2026
N_BOOT = 10_000
SEMILLA = 36


def cargar_predicciones():
    partes = []
    for var in VARIANTES:
        p = pd.read_csv(CARPETA_SALIDA / f'predicciones_{var}.csv')
        p['variante'] = var
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


def error_huerto(g):
    """Mean over years of |total predicted kg - total real kg| / total real kg (in %)."""
    kg_real = (g['real'] * g['has']).groupby(g['anio']).sum()
    kg_pred = (g['pred'] * g['has']).groupby(g['anio']).sum()
    return 100 * ((kg_pred - kg_real).abs() / kg_real).mean()


def metricas(g):
    return pd.Series({
        'MAE': (g['pred'] - g['real']).abs().mean(),
        'MAPE_%': 100 * ((g['pred'] - g['real']).abs() / g['real']).mean(),
        'Error_huerto_%': error_huerto(g),
        'n': len(g),
    })


def por_anio(pred):
    """1. Per-year comparison of the main model against the baseline."""
    var, modelo = PRINCIPAL
    sub = pred[(pred['variante'] == var) & pred['modelo'].isin([BASELINE, modelo])]
    # Group on a copy of the year: pandas excludes grouping keys from the group frame.
    tabla = (sub.assign(anio_grupo=sub['anio'])
             .groupby(['anio_grupo', 'modelo']).apply(metricas).unstack('modelo'))
    tabla.columns = [f'{m}|{c}' for m, c in tabla.columns]
    for m in ['MAE', 'MAPE_%', 'Error_huerto_%']:
        tabla[f'gana_{m}'] = tabla[f'{m}|{modelo}'] < tabla[f'{m}|{BASELINE}']
    return tabla


def baseline_justo(pred):
    """2. Baseline with all past harvests (2019 onward, all cuarteles), on the same test rows."""
    panel = construir_panel()
    var, modelo = PRINCIPAL
    test = pred[(pred['variante'] == var) & (pred['modelo'] == BASELINE)].copy()
    test['pred'] = [
        panel[(panel['anio'] < a) & (panel['cuartel'] == str(c))]['ton_ha'].mean()
        for a, c in zip(test['anio'], test['cuartel'])
    ]
    test['modelo'] = 'B0b Promedio historico (toda la historia)'
    base = pred[(pred['variante'] == var) & pred['modelo'].isin([BASELINE, modelo])]
    return pd.concat([base, test]).groupby('modelo').apply(metricas)


def seleccion_ciega(pred):
    """3. Choose variant x model on 2022-2025 only, then score the choice once on 2026."""
    candidatos = pred[pred['modelo'].isin(MODELOS)]
    sel = (candidatos[candidatos['anio'].isin(ANIOS_SELECCION)]
           .groupby(['variante', 'modelo']).apply(metricas).sort_values('MAE'))
    var, modelo = sel.index[0]
    ciego = pred[(pred['anio'] == ANIO_CIEGO) & (pred['variante'] == var)
                 & pred['modelo'].isin([BASELINE, modelo])]
    return sel, (var, modelo), ciego.groupby('modelo').apply(metricas)


def bootstrap(pred):
    """4. Paired cluster bootstrap of MAE(baseline) - MAE(model), resampling cuarteles."""
    var, modelo = PRINCIPAL
    sub = pred[pred['variante'] == var]
    ancho = sub.pivot_table(index=['cuartel', 'anio'], columns='modelo', values='pred')
    real = sub.drop_duplicates(['cuartel', 'anio']).set_index(['cuartel', 'anio'])['real']
    err_b = (ancho[BASELINE] - real).abs()
    err_m = (ancho[modelo] - real).abs()
    cuarteles = np.array(sorted(real.index.get_level_values('cuartel').unique()))
    grupos = {c: (err_b.xs(c, level='cuartel').values, err_m.xs(c, level='cuartel').values)
              for c in cuarteles}
    rng = np.random.default_rng(SEMILLA)
    difs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        elegidos = rng.choice(cuarteles, size=len(cuarteles), replace=True)
        b = np.concatenate([grupos[c][0] for c in elegidos])
        m = np.concatenate([grupos[c][1] for c in elegidos])
        difs[i] = b.mean() - m.mean()
    observada = err_b.mean() - err_m.mean()
    return {
        'mae_baseline': round(float(err_b.mean()), 3),
        'mae_modelo': round(float(err_m.mean()), 3),
        'diferencia': round(float(observada), 3),
        'reduccion_%': round(float(100 * observada / err_b.mean()), 1),
        'ic95': [round(float(np.percentile(difs, 2.5)), 3), round(float(np.percentile(difs, 97.5)), 3)],
        'prob_mejora': round(float((difs > 0).mean()), 4),
        'n_cuarteles': int(len(cuarteles)),
        'n_predicciones': int(len(real)),
    }


def correlaciones():
    """Coherence with Entrega 1: full-season canopy (leaks harvest months) vs the 31 Jan cutoff."""
    panel = construir_panel()
    salida = {}
    for var in ['cnp_media_completa', 'cnp_acum_ene']:
        d = panel.dropna(subset=[var])
        salida[var] = {'r': round(float(d[var].corr(d['ton_ha'])), 3), 'n': int(len(d))}
    return salida


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    pred = cargar_predicciones()

    print('=== 1. Por anio (variante principal) ===')
    t1 = por_anio(pred)
    print(t1.round(2).T)

    print('\n=== 2. Baseline con toda la historia ===')
    t2 = baseline_justo(pred)
    print(t2.round(2))

    print('\n=== 3. Seleccion ciega (elegir con 2022-2025, probar en 2026) ===')
    sel, elegido, t3 = seleccion_ciega(pred)
    print(sel.head(6).round(2))
    print('Elegido:', elegido)
    print(t3.round(2))

    print('\n=== 4. Bootstrap por cuartel ===')
    t4 = bootstrap(pred)
    print(t4)

    print('\n=== Correlaciones ===')
    t5 = correlaciones()
    print(t5)

    resumen = {
        'por_anio': t1.round(3).reset_index().to_dict(orient='records'),
        'baseline_justo': t2.round(3).reset_index().to_dict(orient='records'),
        'seleccion_top': sel.head(6).round(3).reset_index().to_dict(orient='records'),
        'elegido_ciego': list(elegido),
        'ciego_2026': t3.round(3).reset_index().to_dict(orient='records'),
        'bootstrap': t4,
        'correlaciones': t5,
    }
    (CARPETA_SALIDA / 'robustez.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str),
                                                 encoding='utf-8')
