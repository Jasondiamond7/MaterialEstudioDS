"""Reconstruye el panel cuartel-temporada a partir de los archivos crudos de `Datos kiwi`.

Unidad de analisis: cuartel x anio de cosecha. La temporada agricola va del 1 de mayo
al 30 de abril; la cosecha del anio Y corresponde a la temporada (Y-1)-(Y).

Las metricas de canopia se calculan solo hasta una fecha de corte previa a cosecha
para evitar fuga de informacion (la temporada completa incluye marzo-abril, que es
cosecha y postcosecha).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_DATOS = Path(__file__).resolve().parents[2] / 'Datos kiwi'
CARPETA_SALIDA = Path(__file__).resolve().parent / 'salidas'

# Dias de temporada (1 = 1 de mayo). 154 = 1 de octubre, inicio de brotacion.
DIA_INICIO_CRECIMIENTO = 154
CORTES = {'dic': 245, 'ene': 276, 'feb': 304}


def leer_productivos():
    crudo = pd.read_excel(CARPETA_DATOS / 'COMPARATIVO PRODUCTIVOS.xlsx', header=None)
    anios = crudo.iloc[3, 2:10].astype(int).tolist()
    filas = []
    for i in range(4, len(crudo)):
        cuartel = crudo.iloc[i, 0]
        if pd.isna(cuartel) or str(cuartel).startswith('TOTAL'):
            continue
        has = float(crudo.iloc[i, 1])
        bins = crudo.iloc[i + 1, 2:10].tolist()
        kg = crudo.iloc[i + 2, 2:10].tolist()
        for anio, b, k in zip(anios, bins, kg):
            filas.append({
                'cuartel': str(cuartel).replace(' Y ', '_'),
                'has': has,
                'anio': anio,
                'bins': b,
                'kg_cuartel': k,
                'ton_ha': k / has / 1000,
            })
    return pd.DataFrame(filas)


def leer_canopia():
    canopia = pd.read_excel(
        CARPETA_DATOS / 'CONSOLIDADO_curvas_temporada_planet.xlsx', sheet_name='Datos_diarios',
        usecols=['cuartel', 'fecha', 'temporada', 'dia_temporada', 'cb_invertido_suavizado'],
    )
    canopia['cuartel'] = canopia['cuartel'].str.replace('c', '', regex=False)
    canopia['anio'] = canopia['temporada'].str[-4:].astype(int)
    return canopia


def metricas_canopia(canopia):
    """Metricas por cuartel-anio: temporada completa (como Entrega 1) y hasta cada corte."""
    grupos = canopia.groupby(['cuartel', 'anio'])
    salida = grupos['cb_invertido_suavizado'].mean().rename('cnp_media_completa').to_frame()
    for nombre, dia in CORTES.items():
        # Desde el 1 de mayo hasta el corte: incluye la senal invernal, que es la mas informativa.
        acumulada = canopia[canopia['dia_temporada'] <= dia]
        salida[f'cnp_acum_{nombre}'] = acumulada.groupby(['cuartel', 'anio'])['cb_invertido_suavizado'].mean()
        ventana = canopia[canopia['dia_temporada'].between(DIA_INICIO_CRECIMIENTO, dia)]
        g = ventana.groupby(['cuartel', 'anio'])['cb_invertido_suavizado']
        salida[f'cnp_media_{nombre}'] = g.mean()
        salida[f'cnp_max_{nombre}'] = g.max()
        # Pendiente de crecimiento de la canopia en la ventana (unidades por dia).
        salida[f'cnp_pend_{nombre}'] = ventana.groupby(['cuartel', 'anio'])[
            ['dia_temporada', 'cb_invertido_suavizado']
        ].apply(lambda d: np.polyfit(d['dia_temporada'], d['cb_invertido_suavizado'], 1)[0])
    return salida.reset_index()


def leer_clima():
    eto = pd.read_excel(CARPETA_DATOS / 'datos_eto.xlsx', sheet_name='Calculos_diarios')
    eto['fecha'] = pd.to_datetime(eto['fecha'])
    eto['anio'] = eto['temporada'].str[-4:].astype(int)
    eto['DPV_kPa'] = eto['es_kPa'] - eto['ea_kPa']
    return eto


def metricas_clima(eto, dia_corte=CORTES['ene']):
    """Acumulados por temporada; solo temporadas con invierno completo (cosecha 2023-2026)."""
    eto = eto[eto['anio'].between(2023, 2026)]
    invierno = eto[eto['fecha'].dt.month.isin([5, 6, 7, 8])]
    crecimiento = eto[eto['dia_temporada'].between(DIA_INICIO_CRECIMIENTO - 31, dia_corte)]
    clima = pd.DataFrame({
        'horas_frio': invierno.groupby('anio')['Tmin_C'].apply(lambda t: (t <= 7.2).sum() * 24),
        'gdd_b7': crecimiento.groupby('anio')['Tmean_C'].apply(lambda t: np.maximum(t - 7, 0).sum()),
        'gdd_b10': crecimiento.groupby('anio')['Tmean_C'].apply(lambda t: np.maximum(t - 10, 0).sum()),
        'precip_mm': eto[eto['dia_temporada'] <= dia_corte].groupby('anio')['precip_mm'].sum(),
        'dpv_kpa': crecimiento.groupby('anio')['DPV_kPa'].mean(),
    })
    return clima.reset_index()


def construir_panel():
    productivos = leer_productivos()
    canopia = metricas_canopia(leer_canopia())
    clima = metricas_clima(leer_clima())
    panel = (productivos
             .merge(canopia, on=['cuartel', 'anio'], how='left')
             .merge(clima, on='anio', how='left'))
    return panel.sort_values(['cuartel', 'anio']).reset_index(drop=True)


if __name__ == '__main__':
    CARPETA_SALIDA.mkdir(exist_ok=True)
    panel = construir_panel()
    panel.to_csv(CARPETA_SALIDA / 'panel_integrado.csv', index=False)
    print(panel.shape)
    print(panel.groupby('anio')['ton_ha'].agg(['count', 'mean']).round(1))
    print('Obs con canopia:', panel['cnp_media_ene'].notna().sum())
    print(panel[['ton_ha', 'cnp_media_completa', 'cnp_media_dic', 'cnp_media_ene', 'cnp_media_feb']].corr().round(2))
