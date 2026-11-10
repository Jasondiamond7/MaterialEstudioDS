"""Evidencia preliminar para la propuesta metodologica (Entrega 2).

No es el modelo definitivo: compara baselines y familias de modelos con validacion
temporal expansiva (entrenar con cosechas < Y, predecir Y) para fijar un criterio de
exito realista antes de implementar.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from construir_panel import CARPETA_SALIDA, construir_panel

warnings.filterwarnings('ignore')

ANIOS_TEST = [2022, 2023, 2024, 2025, 2026]
SEMILLA = 36


def predecir_baselines(train, test):
    media_cuartel = train.groupby('cuartel')['ton_ha'].mean()
    ultimo = train.sort_values('anio').groupby('cuartel')['ton_ha'].last()
    return {
        'B0 Promedio historico del cuartel': test['cuartel'].map(media_cuartel).values,
        'B1 Temporada anterior': test['cuartel'].map(ultimo).values,
    }


def matriz(datos, cols, cuarteles):
    x = datos[cols].copy()
    for c in cuarteles[1:]:
        x[f'q_{c}'] = (datos['cuartel'] == c).astype(float)
    return x


def predecir_modelos(train, test, var_cnp):
    cuarteles = sorted(train['cuartel'].unique())
    predicciones = {}

    # Ridge con efecto fijo de cuartel: lineal e interpretable, regularizado por n pequeno.
    cols = [var_cnp]
    xtr, xte = matriz(train, cols, cuarteles), matriz(test, cols, cuarteles)
    media, desv = xtr[cols].mean(), xtr[cols].std()
    xtr[cols], xte[cols] = (xtr[cols] - media) / desv, (xte[cols] - media) / desv
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(xtr, train['ton_ha'])
    predicciones['M1 Ridge (canopia + cuartel)'] = ridge.predict(xte)

    # Modelo mixto: intercepto aleatorio por cuartel (comparte informacion entre cuarteles).
    mixto = smf.mixedlm(f'ton_ha ~ {var_cnp}', train, groups=train['cuartel']).fit(reml=True)
    efectos = {k: v.iloc[0] for k, v in mixto.random_effects.items()}
    predicciones['M2 Modelo mixto (cuartel aleatorio)'] = (
        mixto.predict(test).values + test['cuartel'].map(efectos).fillna(0).values
    )

    # Arboles: el cuartel entra como su promedio historico (target encoding solo con train).
    media_cuartel = train.groupby('cuartel')['ton_ha'].mean()
    cols_arbol = [var_cnp, 'has']
    xtr_a = train[cols_arbol].assign(media_cuartel=train['cuartel'].map(media_cuartel))
    xte_a = test[cols_arbol].assign(media_cuartel=test['cuartel'].map(media_cuartel))
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=SEMILLA)
    predicciones['M3 Random Forest'] = rf.fit(xtr_a, train['ton_ha']).predict(xte_a)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, random_state=SEMILLA)
    predicciones['M4 Gradient Boosting'] = gb.fit(xtr_a, train['ton_ha']).predict(xte_a)
    return predicciones


def validacion_temporal(panel, var_cnp):
    datos = panel.dropna(subset=[var_cnp])
    filas = []
    for anio in ANIOS_TEST:
        train, test = datos[datos['anio'] < anio], datos[datos['anio'] == anio]
        preds = {**predecir_baselines(train, test), **predecir_modelos(train, test, var_cnp)}
        for modelo, pred in preds.items():
            for (_, fila), p in zip(test.iterrows(), pred):
                filas.append({'anio': anio, 'cuartel': fila['cuartel'], 'has': fila['has'],
                              'modelo': modelo, 'real': fila['ton_ha'], 'pred': p})
    return pd.DataFrame(filas)


def resumir(pred):
    def metricas(g):
        kg_real = (g['real'] * g['has']).groupby(g['anio']).sum()
        kg_pred = (g['pred'] * g['has']).groupby(g['anio']).sum()
        return pd.Series({
            'MAPE_%': 100 * mean_absolute_percentage_error(g['real'], g['pred']),
            'RMSE': np.sqrt(mean_squared_error(g['real'], g['pred'])),
            'MAE': mean_absolute_error(g['real'], g['pred']),
            'Error_huerto_%': 100 * ((kg_pred - kg_real).abs() / kg_real).mean(),
        })
    return pred.groupby('modelo').apply(metricas).round(2).sort_values('MAPE_%')


if __name__ == '__main__':
    panel = construir_panel()
    tablas = []
    for var in ['cnp_acum_dic', 'cnp_acum_ene', 'cnp_acum_feb', 'cnp_media_ene', 'cnp_media_completa']:
        pred = validacion_temporal(panel, var)
        pred.to_csv(CARPETA_SALIDA / f'predicciones_{var}.csv', index=False)
        tabla = resumir(pred)
        tabla.insert(0, 'variable_canopia', var)
        tablas.append(tabla)
        print(f'\n=== {var} ===')
        print(tabla)
        print('MAPE por anio:')
        print(pred.assign(ape=100 * (pred['pred'] - pred['real']).abs() / pred['real'])
              .pivot_table(index='modelo', columns='anio', values='ape', aggfunc='mean').round(1))
    pd.concat(tablas).to_csv(CARPETA_SALIDA / 'resultados_validacion.csv')
