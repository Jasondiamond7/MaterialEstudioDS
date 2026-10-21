# -*- coding: utf-8 -*-
"""
Generador de `build_notebook_v2.py`.

Lee el notebook original (Grupo_36_Taller3.ipynb.bak), aplica un conjunto
determinista de parches a las celdas de código (para publicar los valores con
`_pub`) y a las celdas markdown (para sustituir números por marcadores
`{{clave}}`), y escribe `build_notebook_v2.py` con la lista `CELLS`.

Los parches son estrictos: cada texto ancla debe aparecer EXACTAMENTE una vez
en la celda destino; si no, el generador falla con un mensaje explícito.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
ORIGEN = BASE.parent / "Grupo_36_version2" / "Grupo_36_Taller3.ipynb.bak"
DESTINO = BASE / "build_notebook_v2.py"

# --------------------------------------------------------------------------
# PARCHES DE CÓDIGO: (indice_celda, texto_original, texto_nuevo)
# --------------------------------------------------------------------------
CODE_PATCHES: dict[int, list[tuple[str, str]]] = {}

CODE_PATCHES[7] = [
    (
        """SEMILLA = 2026
np.random.seed(SEMILLA)
CARPETA = Path('.')""",
        """SEMILLA = 2026
np.random.seed(SEMILLA)

R = {}   # valores que alimentan el texto del informe


def _pub(clave, valor):
    \"\"\"Registra un valor para el texto y lo publica en formato parseable.\"\"\"
    R[clave] = valor
    print(f'@@{clave}={valor}@@')


def _num(x, dec=4):
    \"\"\"Formatea un número al estilo español (coma decimal).\"\"\"
    return f'{float(x):.{dec}f}'.replace('.', ',')


CARPETA = Path('.')""",
    ),
]

CODE_PATCHES[11] = [
    (
        """print('Respuesta:', RESPUESTA, '-> valores', sorted(entrenamiento[RESPUESTA].unique()))
# duracion se conoce solo al terminar la llamada; su relación con la respuesta lo muestra:
print(entrenamiento.groupby(RESPUESTA)['duracion'].describe()[['mean', '50%']])""",
        """print('Respuesta:', RESPUESTA, '-> valores', sorted(entrenamiento[RESPUESTA].unique()))
# duracion se conoce solo al terminar la llamada; su relación con la respuesta lo muestra:
print(entrenamiento.groupby(RESPUESTA)['duracion'].describe()[['mean', '50%']])

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('n_registros_entrenamiento', len(entrenamiento))
_pub('n_predictores_originales', len(predictores_originales))
_pub('duracion_media_contrata', _num(entrenamiento.loc[entrenamiento[RESPUESTA] == 1, 'duracion'].mean(), 0))
_pub('duracion_media_no_contrata', _num(entrenamiento.loc[entrenamiento[RESPUESTA] == 0, 'duracion'].mean(), 0))""",
    ),
]

CODE_PATCHES[14] = [
    (
        """print('Balance por pliegue:'); print(entrenamiento.groupby('pliegue')[RESPUESTA].agg(['size', 'mean']).round(4))""",
        """print('Balance por pliegue:'); print(entrenamiento.groupby('pliegue')[RESPUESTA].agg(['size', 'mean']).round(4))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('n_numericas', len(numericas))
_pub('n_categoricas', len(categoricas))
_pub('unk_resultado_previo_pct', _num(100 * (entrenamiento['resultado_previo'] == 'unknown').mean(), 1))
_pub('unk_canal_pct', _num(100 * (entrenamiento['canal'] == 'unknown').mean(), 1))
_pub('unk_educacion_pct', _num(100 * (entrenamiento['educacion'] == 'unknown').mean(), 1))
_pub('unk_ocupacion_pct', _num(100 * (entrenamiento['ocupacion'] == 'unknown').mean(), 1))
_pub('n_duplicados', int(entrenamiento.drop(columns=['id_cliente', 'pliegue']).duplicated().sum()))
_pub('n_clase0', int((entrenamiento[RESPUESTA] == 0).sum()))
_pub('n_clase1', int((entrenamiento[RESPUESTA] == 1).sum()))
_pub('prop_clase0', _num(100 * (entrenamiento[RESPUESTA] == 0).mean(), 1))
_pub('prop_clase1', _num(100 * (entrenamiento[RESPUESTA] == 1).mean(), 1))
_pub('n_saldo_neg', int((entrenamiento.saldo < 0).sum()))
_pub('min_saldo', int(entrenamiento.saldo.min()))
_pub('n_dias_m1', int((entrenamiento.dias_previos == -1).sum()))
_pub('acc_mayoritaria', _num(100 * acc_mayoritaria, 1))""",
    ),
]

CODE_PATCHES[17] = [
    (
        """print('\\nNuméricas estandarizadas (media ~0, desv ~1):')
print(pd.DataFrame(X_demo[:, :len(NUM)], columns=NUM).agg(['mean', 'std']).round(3))""",
        """print('\\nNuméricas estandarizadas (media ~0, desv ~1):')
print(pd.DataFrame(X_demo[:, :len(NUM)], columns=NUM).agg(['mean', 'std']).round(3))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('tasa_contactado_si', _num(100 * datos.loc[datos.contactado_antes == 1, RESPUESTA].mean(), 1))
_pub('tasa_contactado_no', _num(100 * datos.loc[datos.contactado_antes == 0, RESPUESTA].mean(), 1))
_pub('skew_contactos_previos', _num(entrenamiento.contactos_previos.skew(), 2))
_pub('max_contactos_previos', int(entrenamiento.contactos_previos.max()))
_pub('skew_log_contactos', _num(datos.log_contactos_previos.skew(), 2))
_pub('p_columnas', len(COLUMNAS))
_pub('n_num_preparadas', len(NUM))
_pub('n_bin_preparadas', len(BIN))
_pub('n_dummies', len(COLUMNAS) - len(NUM) - len(BIN))
_pub('saldo_desv', _num(entrenamiento.saldo.std(ddof=1), 0))""",
    ),
]

CODE_PATCHES[22] = [
    (
        """print()
print(pd.DataFrame({k: metricas(y, oof[k]) for k in oof}).T.round(4))""",
        """print()
mt = pd.DataFrame({k: metricas(y, oof[k]) for k in oof}).T.round(4)
print(mt)

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('log_todas_log_loss', _num(mt.loc['Logística (todas)', 'log_loss']))
_pub('log_todas_brier', _num(mt.loc['Logística (todas)', 'brier']))
_pub('log_todas_auc', _num(mt.loc['Logística (todas)', 'auc']))
_pub('log10_log_loss', _num(mt.loc['Logística (10 F-ANOVA)', 'log_loss']))
_pub('log10_brier', _num(mt.loc['Logística (10 F-ANOVA)', 'brier']))
_pub('log10_auc', _num(mt.loc['Logística (10 F-ANOVA)', 'auc']))
_pub('delta_auc_logistica', _num(mt.loc['Logística (todas)', 'auc'] - mt.loc['Logística (10 F-ANOVA)', 'auc']))
_pub('sel_mes_mar_folds', int(frec.get('mes_mar', 0)))
_pub('sel_estado_civil_married_folds', int(frec.get('estado_civil_married', 0)))
_pub('sel_n_todas', int((frec == 5).sum()))
_pub('sel_pliegue_estado_civil', int(max(f for f, c in seleccion.items() if 'estado_civil_married' in c)))""",
    ),
]

CODE_PATCHES[25] = [
    (
        """MODELO_ELEGIDO = resultados['log_loss'].idxmin()
print('\\nModelo probabilístico con menor log loss OOF:', MODELO_ELEGIDO)""",
        """MODELO_ELEGIDO = resultados['log_loss'].idxmin()
print('\\nModelo probabilístico con menor log loss OOF:', MODELO_ELEGIDO)

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
def _v(nombre, col):
    return _num(resultados.loc[nombre, col])


_pub('modelo_elegido', MODELO_ELEGIDO)
_pub('modelo_elegido_nombre', MODELO_ELEGIDO.split(' (')[0])
_pub('gb_log_loss', _v('Gradient Boosting (lr=0.05)', 'log_loss'))
_pub('gb_brier', _v('Gradient Boosting (lr=0.05)', 'brier'))
_pub('gb_auc', _v('Gradient Boosting (lr=0.05)', 'auc'))
_pub('rf_log_loss', _v('Random Forest', 'log_loss'))
_pub('rf_brier', _v('Random Forest', 'brier'))
_pub('rf_auc', _v('Random Forest', 'auc'))
_pub('log_todas_log_loss', _v('Logística (todas)', 'log_loss'))
_pub('log_todas_brier', _v('Logística (todas)', 'brier'))
_pub('log_todas_auc', _v('Logística (todas)', 'auc'))
_pub('log10_log_loss', _v('Logística (10 F-ANOVA)', 'log_loss'))
_pub('log10_brier', _v('Logística (10 F-ANOVA)', 'brier'))
_pub('log10_auc', _v('Logística (10 F-ANOVA)', 'auc'))
_pub('arb3_log_loss', _v('Árbol (prof. 3)', 'log_loss'))
_pub('arb3_brier', _v('Árbol (prof. 3)', 'brier'))
_pub('arb3_auc', _v('Árbol (prof. 3)', 'auc'))
_pub('arb3_log_loss_train', _v('Árbol (prof. 3)', 'log_loss_entren_medio'))
_pub('arb6_log_loss', _v('Árbol (prof. 6)', 'log_loss'))
_pub('arb6_brier', _v('Árbol (prof. 6)', 'brier'))
_pub('arb6_auc', _v('Árbol (prof. 6)', 'auc'))
_pub('arb6_log_loss_train', _v('Árbol (prof. 6)', 'log_loss_entren_medio'))
_pub('ref_log_loss', _v('Referencia (proporción)', 'log_loss'))
_pub('ref_brier', _v('Referencia (proporción)', 'brier'))
_pub('ref_auc', _v('Referencia (proporción)', 'auc'))
_pub('nb_log_loss', _v('Naive Bayes gaussiano', 'log_loss'))
_pub('nb_brier', _v('Naive Bayes gaussiano', 'brier'))
_pub('nb_auc', _v('Naive Bayes gaussiano', 'auc'))
_pub('svm_auc', _num(roc_auc_score(y, oof['SVM radial (C=1, γ=1/p)'])))
_pub('delta_ll_gb_rf', _num(resultados.loc['Random Forest', 'log_loss'] - resultados.loc['Gradient Boosting (lr=0.05)', 'log_loss']))
_pub('delta_ll_gb_log', _num(resultados.loc['Logística (todas)', 'log_loss'] - resultados.loc['Gradient Boosting (lr=0.05)', 'log_loss']))
_pub('mejora_pct_gb', _num(100 * (resultados.loc['Referencia (proporción)', 'log_loss'] - resultados.loc['Gradient Boosting (lr=0.05)', 'log_loss']) / resultados.loc['Referencia (proporción)', 'log_loss'], 0))""",
    ),
]

CODE_PATCHES[28] = [
    (
        """print('SVM sin estandarizar numéricas: AUC OOF = %.4f' % roc_auc_score(y, validar_svm_sin_escala()))""",
        """auc_sin_escala = roc_auc_score(y, validar_svm_sin_escala())
print('SVM sin estandarizar numéricas: AUC OOF = %.4f' % auc_sin_escala)

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
gr_pivot = pd.DataFrame(grilla).pivot(index='C', columns='gamma', values='auc')
_pub('arb3_brecha', _num(arb.loc['Árbol (prof. 3)', 'brecha']))
_pub('arb6_brecha', _num(arb.loc['Árbol (prof. 6)', 'brecha']))
_pub('nb_frac_bajo001', _num(100 * (oof['Naive Bayes gaussiano'] < .01).mean(), 0))
_pub('nb_frac_sobre099', _num(100 * (oof['Naive Bayes gaussiano'] > .99).mean(), 0))
_pub('nb_media', _num(oof['Naive Bayes gaussiano'].mean(), 2))
_pub('tasa_real', _num(100 * y.mean(), 1))
_pub('svm_auc_sin_escala', _num(auc_sin_escala))
_pub('svm_auc_g01_min', _num(gr_pivot['0.1/p'].min()))
_pub('svm_auc_g01_max', _num(gr_pivot['0.1/p'].max()))
_pub('svm_auc_g1_min', _num(gr_pivot['1/p'].min()))
_pub('svm_auc_g1_max', _num(gr_pivot['1/p'].max()))
_pub('svm_auc_g10_min', _num(gr_pivot['10/p'].min()))
_pub('svm_auc_g10_max', _num(gr_pivot['10/p'].max()))
_pub('sqrt_p', _num(np.sqrt(len(COLUMNAS)), 1))
_pub('n_candidatas_rf', int(np.sqrt(len(COLUMNAS))))""",
    ),
]

CODE_PATCHES[32] = [
    (
        """UMBRAL = float(tabla_costos.loc[tabla_costos.costo.idxmin(), 'umbral'])
print(f'\\nUmbral elegido (mínimo costo OOF): {UMBRAL}')
print('Costo de no contactar a nadie (todo FN): %d | de contactar a todos (todo FP): %d' % (5 * y.sum(), (1 - y).sum()))""",
        """UMBRAL = float(tabla_costos.loc[tabla_costos.costo.idxmin(), 'umbral'])
print(f'\\nUmbral elegido (mínimo costo OOF): {UMBRAL}')
print('Costo de no contactar a nadie (todo FN): %d | de contactar a todos (todo FP): %d' % (5 * y.sum(), (1 - y).sum()))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
def _fila(t):
    return tabla_costos.loc[tabla_costos.umbral == t].iloc[0]


for _t, _pre in [(0.05, '005'), (0.10, '010'), (0.15, '015'), (0.20, '020'), (0.25, '025'), (0.50, '050')]:
    _f = _fila(_t)
    _pub(f'umbral_{_pre}', _num(_t, 2))
    _pub(f'costo_{_pre}', int(_f.costo))
    _pub(f'fp_{_pre}', int(_f.FP))
    _pub(f'fn_{_pre}', int(_f.FN))
_f = _fila(UMBRAL)
_pub('umbral_elegido', _num(UMBRAL, 2))
_pub('costo_umbral', int(_f.costo))
_pub('costo_n_umbral', _num(_f.costo / len(y)))
_pub('tp_umbral', int(_f.TP))
_pub('fp_umbral', int(_f.FP))
_pub('fn_umbral', int(_f.FN))
_pub('tn_umbral', int(_f.TN))
_pub('sens_umbral', _num(100 * _f.sensibilidad, 0))
_pub('espec_umbral', _num(100 * _f.especificidad, 1))
_pub('contactados_umbral', int(_f.contactados))
_pub('pct_contactados_umbral', _num(100 * _f.contactados / len(y), 0))
_pub('costo_no_contactar', int(5 * y.sum()))
_pub('costo_contactar_todos', int((1 - y).sum()))
_pub('n_positivos', int(y.sum()))
_pub('sens_umbral_050', _num(100 * _fila(0.50).sensibilidad, 0))""",
    ),
]

CODE_PATCHES[35] = [
    (
        """dec = pd.qcut(pd.Series(p_test).rank(method='first', ascending=False), 10, labels=range(1, 11))
print('\\nTasa real de contratación por decil de probabilidad (1 = más alta):')
print(pd.DataFrame({'y': y_test, 'decil': dec}).groupby('decil', observed=True)['y'].agg(['size', 'mean']).round(3).T)""",
        """dec = pd.qcut(pd.Series(p_test).rank(method='first', ascending=False), 10, labels=range(1, 11))
por_decil = pd.DataFrame({'y': y_test, 'decil': dec}).groupby('decil', observed=True)['y'].agg(['size', 'mean']).round(3)
print('\\nTasa real de contratación por decil de probabilidad (1 = más alta):')
print(por_decil.T)

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_g15 = comp.iloc[0]
_g50 = comp.iloc[1]
_ref = comp.iloc[2]
_todos = comp.iloc[4]
_pub('n_test', len(y_test))
_pub('tasa_test', _num(100 * y_test.mean(), 2))
_pub('p_ref', _num(y.mean(), 4))
_pub('gb_test_log_loss', _num(met.loc[MODELO_ELEGIDO, 'log_loss']))
_pub('gb_test_brier', _num(met.loc[MODELO_ELEGIDO, 'brier']))
_pub('gb_test_auc', _num(met.loc[MODELO_ELEGIDO, 'auc']))
_pub('ref_test_log_loss', _num(met.loc['Referencia', 'log_loss']))
_pub('ref_test_brier', _num(met.loc['Referencia', 'brier']))
_pub('ref_test_auc', _num(met.loc['Referencia', 'auc']))
_pub('test_tp_015', int(_g15.TP))
_pub('test_fp_015', int(_g15.FP))
_pub('test_fn_015', int(_g15.FN))
_pub('test_tn_015', int(_g15.TN))
_pub('test_sens_015', _num(100 * _g15.sensibilidad, 1))
_pub('test_espec_015', _num(100 * _g15.especificidad, 1))
_pub('test_contactados_015', int(_g15.contactados))
_pub('test_costo_015', int(_g15.C))
_pub('test_cn_015', _num(_g15['C/n']))
_pub('test_tp_050', int(_g50.TP))
_pub('test_fp_050', int(_g50.FP))
_pub('test_fn_050', int(_g50.FN))
_pub('test_tn_050', int(_g50.TN))
_pub('test_sens_050', _num(100 * _g50.sensibilidad, 1))
_pub('test_espec_050', _num(100 * _g50.especificidad, 1))
_pub('test_contactados_050', int(_g50.contactados))
_pub('test_costo_050', int(_g50.C))
_pub('test_cn_050', _num(_g50['C/n']))
_pub('test_costo_ref', int(_ref.C))
_pub('test_cn_ref', _num(_ref['C/n']))
_pub('test_costo_todos', int(_todos.C))
_pub('test_cn_todos', _num(_todos['C/n']))
_pub('test_reduccion_pct_050', _num(100 * (_g50.C - _g15.C) / _g50.C, 0))
_pub('test_reduccion_pct_ref', _num(100 * (_ref.C - _g15.C) / _ref.C, 0))
_pub('test_n_positivos', int(y_test.sum()))
_pub('test_n_negativos', int((y_test == 0).sum()))
_pub('test_n_test_menos', int(len(y_test) - y_test.sum()))
_pub('test_pct_cartera', _num(100 * _g15.contactados / len(y_test), 0))
_pub('test_tasa_exito', _num(100 * _g15.TP / _g15.contactados, 0))
_pub('test_veces_tasa_base', _num((_g15.TP / _g15.contactados) / y_test.mean(), 1))
_pub('test_decil1_tasa', _num(100 * por_decil.loc[1, 'mean'], 0))
_pub('test_se_sens', _num(100 * np.sqrt(_g15.sensibilidad * (1 - _g15.sensibilidad) / int(y_test.sum())), 1))""",
    ),
]

CODE_PATCHES[39] = [
    (
        """print('\\nCorrelaciones entre las 4 variables:'); print(Z.corr().round(3))""",
        """print('\\nCorrelaciones entre las 4 variables:'); print(Z.corr().round(3))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
for _pc in var.index:
    _pub(f'pca_{_pc.lower()}_var', _num(var.loc[_pc, 'varianza']))
    _pub(f'pca_{_pc.lower()}_pct', _num(100 * var.loc[_pc, 'prop_explicada'], 1))
    _pub(f'pca_{_pc.lower()}_acum', _num(100 * var.loc[_pc, 'acumulada'], 1))
_pub('pca_sum_pct', _num(100 * pca.explained_variance_ratio_[:2].sum(), 1))
_pub('pca_omitido_pct', _num(100 * (1 - pca.explained_variance_ratio_[:2].sum()), 1))
for _v in VARS4:
    _pub('pca_carga_' + _v + '_pc1', f'{cargas.loc[_v, "PC1"]:+.2f}'.replace('.', ','))
    _pub('pca_carga_' + _v + '_pc2', f'{cargas.loc[_v, "PC2"]:+.2f}'.replace('.', ','))
_corr = Z.corr().to_numpy()
_abs = np.abs(_corr).copy()
np.fill_diagonal(_abs, 0.0)
_i, _j = np.unravel_index(int(np.argmax(_abs)), _abs.shape)
_pub('pca_max_corr', _num(_corr[_i, _j], 3))
_pub('pca_max_corr_par', f'{VARS4[_i]} y {VARS4[_j]}')""",
    ),
]

CODE_PATCHES[42] = [
    (
        """ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(f'K-means (k = {K}) en el plano PCA'); ax.legend()
plt.tight_layout(); plt.show()""",
        """ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(f'K-means (k = {K}) en el plano PCA'); ax.legend()
plt.tight_layout(); plt.show()

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
for _i, _k in enumerate([2, 3, 4]):
    _fila_k = resumen_k.loc[_i]
    _pub(f'k{_k}_wss', f'{_fila_k.WSS:,.0f}'.replace(',', ' '))
    _pub(f'k{_k}_sil', _num(_fila_k.silueta_media, 3))
    _pub(f'k{_k}_tamanos', ' / '.join(str(int(_t)) for _t in _fila_k['tamaños']))
_pub('k_elegido', int(K))
_pub('silueta_k', _num(resumen_k.loc[resumen_k.silueta_media.idxmax(), 'silueta_media'], 3))
for _g in range(K):
    _p = perfil.loc[_g]
    _pub(f'g{_g}_n', int(_p.n))
    _pub(f'g{_g}_edad', _num(_p.edad, 0))
    _pub(f'g{_g}_saldo', _num(_p.saldo, 0))
    _pub(f'g{_g}_saldo_med', _num(_p.saldo_mediana, 0))
    _pub(f'g{_g}_contactos', _num(_p.contactos_campana, 1))
    _pub(f'g{_g}_contactos_previos', _num(_p.contactos_previos, 1))
    _pub(f'g{_g}_pct_previo', _num(100 * _p.pct_con_contacto_previo, 0))
    _pub(f'g{_g}_tasa', _num(100 * _p.tasa_contratacion, 1))
_pub('tasa_global', _num(100 * entrenamiento[RESPUESTA].mean(), 1))
_pub('k_max_tasa', _num(100 * perfil.tasa_contratacion.max(), 1))""",
    ),
]

CODE_PATCHES[46] = [
    (
        """print('\\nValores estandarizados de esos registros:')
print(Z.loc[top2.index].round(2).assign(id_cliente=top2.id_cliente.values).to_string(index=False))""",
        """print('\\nValores estandarizados de esos registros:')
print(Z.loc[top2.index].round(2).assign(id_cliente=top2.id_cliente.values).to_string(index=False))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('corte_chi2', _num(corte, 4))
_pub('n_marcados', int(marcados.sum()))
_pub('pct_marcados', _num(100 * marcados.mean(), 2))
for _i, (_idx, _row) in enumerate(top2.iterrows()):
    _pub(f'anom{_i + 1}_id', int(_row.id_cliente))
    _pub(f'anom{_i + 1}_edad', int(_row.edad))
    _pub(f'anom{_i + 1}_saldo', f'{int(_row.saldo):,}'.replace(',', ' '))
    _pub(f'anom{_i + 1}_contactos_campana', int(_row.contactos_campana))
    _pub(f'anom{_i + 1}_contactos_previos', int(_row.contactos_previos))
    _pub(f'anom{_i + 1}_d2', _num(_row.d2, 1))
    _pub(f'anom{_i + 1}_contrata', int(_row.contrata))""",
    ),
]

CODE_PATCHES[49] = [
    (
        """print('\\nTasa de contratación: marcados %.3f vs no marcados %.3f' % (entrenamiento.contrata[marcados].mean(), entrenamiento.contrata[~marcados].mean()))""",
        """print('\\nTasa de contratación: marcados %.3f vs no marcados %.3f' % (entrenamiento.contrata[marcados].mean(), entrenamiento.contrata[~marcados].mean()))

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('saldo_mediana', f'{int(entrenamiento.saldo.median()):,}'.replace(',', ' '))
_pub('saldo_p99', f'{int(np.percentile(entrenamiento.saldo, 99)):,}'.replace(',', ' '))
_pub('cc_mediana', f'{int(entrenamiento.contactos_campana.median()):,}'.replace(',', ' '))
_pub('cc_p99', f'{int(np.percentile(entrenamiento.contactos_campana, 99)):,}'.replace(',', ' '))
for _i, _idx in enumerate(top2.index):
    _pub(f'z{_i + 1}_saldo', _num(Z.loc[_idx, 'saldo'], 2))
    _pub(f'z{_i + 1}_contactos_campana', _num(Z.loc[_idx, 'contactos_campana'], 2))
    _pub(f'd2_{_i + 1}', _num(top2.d2.iloc[_i], 1))
_pub('skew_saldo', _num(entrenamiento.saldo.skew(), 2))
_pub('skew_contactos_campana', _num(entrenamiento.contactos_campana.skew(), 2))
_pub('skew_contactos_previos', _num(entrenamiento.contactos_previos.skew(), 2))
_pub('kurt_saldo', _num(entrenamiento.saldo.kurt(), 0))
_pub('kurt_contactos_campana', _num(entrenamiento.contactos_campana.kurt(), 0))
_pub('kurt_contactos_previos', _num(entrenamiento.contactos_previos.kurt(), 0))
_pub('factor_exceso', _num(100 * marcados.mean() / 1.0, 1))
_pub('tasa_marcados', _num(100 * entrenamiento.contrata[marcados].mean(), 1))
_pub('tasa_no_marcados', _num(100 * entrenamiento.contrata[~marcados].mean(), 1))""",
    ),
]

# --------------------------------------------------------------------------
# PARCHES DE MARKDOWN: (indice_celda, texto_original, texto_nuevo)
# --------------------------------------------------------------------------
MD_PATCHES: dict[int, list[tuple[str, str]]] = {}

MD_PATCHES[12] = [
    (
        "hay 3617 en entrenamiento.",
        "hay {{n_registros_entrenamiento}} en entrenamiento.",
    ),
    (
        "la duración media es 554 s en quienes contratan vs 228 s en quienes no",
        "la duración media es {{duracion_media_contrata}} s en quienes contratan vs {{duracion_media_no_contrata}} s en quienes no",
    ),
    (
        "Quedan 15 predictores originales.",
        "Quedan {{n_predictores_originales}} predictores originales.",
    ),
]

MD_PATCHES[15] = [
    (
        "**Tipos:** 6 numéricas enteras",
        "**Tipos:** {{n_numericas}} numéricas enteras",
    ),
    (
        "y 9 categóricas (`ocupacion`",
        "y {{n_categoricas}} categóricas (`ocupacion`",
    ),
    (
        "`resultado_previo` 82,4 % (no es un faltante real",
        "`resultado_previo` {{unk_resultado_previo_pct}} % (no es un faltante real",
    ),
    (
        "`canal` 30,1 %",
        "`canal` {{unk_canal_pct}} %",
    ),
    (
        "`educacion` 4,3 %",
        "`educacion` {{unk_educacion_pct}} %",
    ),
    (
        "`ocupacion` 1,0 %",
        "`ocupacion` {{unk_ocupacion_pct}} %",
    ),
    (
        "**Duplicados:** 0 filas duplicadas",
        "**Duplicados:** {{n_duplicados}} filas duplicadas",
    ),
    (
        "**Balance:** 3200 no contratan (88,5 %) y 417 contratan (11,5 %)",
        "**Balance:** {{n_clase0}} no contratan ({{prop_clase0}} %) y {{n_clase1}} contratan ({{prop_clase1}} %)",
    ),
    (
        "(302 registros, mínimo −3313 €)",
        "({{n_saldo_neg}} registros, mínimo {{min_saldo}} €)",
    ),
    (
        "**`dias_previos` = −1** (2979 registros)",
        "**`dias_previos` = −1** ({{n_dias_m1}} registros)",
    ),
    (
        "logra una exactitud de **88,5 %**",
        "logra una exactitud de **{{acc_mayoritaria}} %**",
    ),
]

MD_PATCHES[18] = [
    (
        "la tasa de contratación es 22,4 % entre los contactados antes vs 9,2 % en el resto",
        "la tasa de contratación es {{tasa_contactado_si}} % entre los contactados antes vs {{tasa_contactado_no}} % en el resto",
    ),
    (
        "(asimetría 5,85, máximo 25)",
        "(asimetría {{skew_contactos_previos}}, máximo {{max_contactos_previos}})",
    ),
    (
        "el logaritmo comprime la cola (asimetría 2,51)",
        "el logaritmo comprime la cola (asimetría {{skew_log_contactos}})",
    ),
    (
        "- Resultado: **p = 42 columnas** (6 numéricas, 1 indicador y 35 dummies).",
        "- Resultado: **p = {{p_columnas}} columnas** ({{n_num_preparadas}} numéricas, {{n_bin_preparadas}} indicador y {{n_dummies}} dummies).",
    ),
    (
        "sin escalar `saldo` (desv. ≈ 3000)",
        "sin escalar `saldo` (desv. ≈ {{saldo_desv}})",
    ),
]

MD_PATCHES[23] = [
    (
        "`mes_oct` (5/5) y `mes_mar` (4/5); en el pliegue 5 entra `estado_civil_married` en lugar de `mes_mar`.",
        "`mes_oct` (las {{sel_n_todas}} anteriores aparecen en los 5 pliegues) y `mes_mar` ({{sel_mes_mar_folds}}/5); en el pliegue {{sel_pliegue_estado_civil}} entra `estado_civil_married` en lugar de `mes_mar`.",
    ),
    (
        "**redundante**: cuatro columnas (`dias_previos`",
        "**redundante**: varias de las columnas seleccionadas (`dias_previos`",
    ),
    (
        "| Logística, 42 columnas | **0,3151** | **0,0895** | **0,7250** |",
        "| Logística, 42 columnas | **{{log_todas_log_loss}}** | **{{log_todas_brier}}** | **{{log_todas_auc}}** |",
    ),
    (
        "| Logística, 10 columnas F | 0,3203 | 0,0914 | 0,7011 |",
        "| Logística, 10 columnas F | {{log10_log_loss}} | {{log10_brier}} | {{log10_auc}} |",
    ),
    (
        "a costo de ~0,024 de AUC.",
        "a costo de ~{{delta_auc_logistica}} de AUC.",
    ),
    (
        "que con 3617 observaciones el modelo completo",
        "que con {{n_registros_entrenamiento}} observaciones el modelo completo",
    ),
]

MD_PATCHES[26] = [
    (
        "| **Gradient Boosting** (lr = 0,05; 200 árboles; prof. 3) | **0,3110** | **0,0895** | **0,7443** | – |",
        "| **Gradient Boosting** (lr = 0,05; 200 árboles; prof. 3) | **{{gb_log_loss}}** | **{{gb_brier}}** | **{{gb_auc}}** | – |",
    ),
    (
        "| Random Forest | 0,3122 | 0,0898 | 0,7400 | – |",
        "| Random Forest | {{rf_log_loss}} | {{rf_brier}} | {{rf_auc}} | – |",
    ),
    (
        "| Logística (42 columnas) | 0,3151 | 0,0895 | 0,7250 | – |",
        "| Logística (42 columnas) | {{log_todas_log_loss}} | {{log_todas_brier}} | {{log_todas_auc}} | – |",
    ),
    (
        "| Logística (10 F) | 0,3203 | 0,0914 | 0,7011 | – |",
        "| Logística (10 F) | {{log10_log_loss}} | {{log10_brier}} | {{log10_auc}} | – |",
    ),
    (
        "| Árbol prof. 3 | 0,3298 | 0,0932 | 0,6363 | 0,3199 |",
        "| Árbol prof. 3 | {{arb3_log_loss}} | {{arb3_brier}} | {{arb3_auc}} | {{arb3_log_loss_train}} |",
    ),
    (
        "| Árbol prof. 6 | 0,3393 | 0,0921 | 0,6636 | 0,3019 |",
        "| Árbol prof. 6 | {{arb6_log_loss}} | {{arb6_brier}} | {{arb6_auc}} | {{arb6_log_loss_train}} |",
    ),
    (
        "| Referencia (proporción) | 0,3574 | 0,1020 | 0,50 | – |",
        "| Referencia (proporción) | {{ref_log_loss}} | {{ref_brier}} | {{ref_auc}} | – |",
    ),
    (
        "| Naive Bayes gaussiano | 2,2559 | 0,1648 | 0,7024 | – |",
        "| Naive Bayes gaussiano | {{nb_log_loss}} | {{nb_brier}} | {{nb_auc}} | – |",
    ),
    (
        "| SVM radial (C = 1, γ = 1/42) | – | – | 0,6851 | – |",
        "| SVM radial (C = 1, γ = 1/{{p_columnas}}) | – | – | {{svm_auc}} | – |",
    ),
    (
        "**Modelo seleccionado: Gradient Boosting**, con el menor log loss OOF (0,3110).",
        "**Modelo seleccionado: {{modelo_elegido}}**, con el menor log loss OOF ({{gb_log_loss}}).",
    ),
    (
        "También tiene el mejor AUC y empata en Brier con la logística.",
        "También tiene el mejor AUC y un Brier prácticamente igual al de la logística ({{gb_brier}} frente a {{log_todas_brier}}).",
    ),
    (
        "son pequeñas (≈ 0,001–0,004 en log loss)",
        "son pequeñas (≈ {{delta_ll_gb_rf}}–{{delta_ll_gb_log}} en log loss)",
    ),
    (
        "moderada (0,357 → 0,311, ≈ 13 %)",
        "moderada ({{ref_log_loss}} → {{gb_log_loss}}, ≈ {{mejora_pct_gb}} %)",
    ),
    (
        "de manera moderada (AUC ≈ 0,74)",
        "de manera moderada (AUC ≈ {{gb_auc}})",
    ),
]

MD_PATCHES[29] = [
    (
        "(0,302 vs 0,320) pero **peor** log loss OOF (0,339 vs 0,330)",
        "({{arb6_log_loss_train}} vs {{arb3_log_loss_train}}) pero **peor** log loss OOF ({{arb6_log_loss}} vs {{arb3_log_loss}})",
    ),
    (
        "crece de 0,010 a 0,037.",
        "crece de {{arb3_brecha}} a {{arb6_brecha}}.",
    ),
    (
        "ordena algo mejor (AUC 0,664 vs 0,636)",
        "ordena algo mejor (AUC {{arb6_auc}} vs {{arb3_auc}})",
    ),
    (
        "añade la selección aleatoria de √p ≈ 6 columnas candidatas por división",
        "añade la selección aleatoria de √p columnas candidatas por división (con p = {{p_columnas}}, son {{n_candidatas_rf}} candidatas)",
    ),
    (
        "el 75 % de las probabilidades OOF son < 0,01 y el 11 % > 0,99, y la media es 0,16 (tasa real 0,115)",
        "el {{nb_frac_bajo001}} % de las probabilidades OOF son < 0,01 y el {{nb_frac_sobre099}} % > 0,99, y la media es {{nb_media}} (tasa real {{tasa_real}} %)",
    ),
    (
        "**mal calibrado** (log loss 2,26, peor que la referencia) aunque ordena de forma razonable (AUC 0,70)",
        "**mal calibrado** (log loss {{nb_log_loss}}, peor que la referencia) aunque ordena de forma razonable (AUC {{nb_auc}})",
    ),
    (
        "γ = 0,1/p da los peores resultados (0,63–0,68) y γ = 1/p o 10/p mejoran (0,69–0,70)",
        "γ = 0,1/p da los peores resultados ({{svm_auc_g01_min}}–{{svm_auc_g01_max}}) y γ = 1/p o 10/p mejoran ({{svm_auc_g1_min}}–{{svm_auc_g1_max}} y {{svm_auc_g10_min}}–{{svm_auc_g10_max}})",
    ),
    (
        "el AUC cae de 0,685 a **0,599**",
        "el AUC cae de {{svm_auc}} a **{{svm_auc_sin_escala}}**",
    ),
    (
        "Naive Bayes tiene AUC 0,70 (similar a la logística con 10 columnas)",
        "Naive Bayes tiene AUC {{nb_auc}} (similar a la logística con 10 columnas, {{log10_auc}})",
    ),
]

MD_PATCHES[33] = [
    (
        "Con las probabilidades OOF del Gradient Boosting, el costo C(t) = FP(t) + 5·FN(t) se minimiza en **t = 0,15** (C = 1492; C/n = 0,412), con 192 TP, 367 FP, 225 FN y 2833 TN: sensibilidad 46 % y especificidad 88,5 %, contactando a 559 de 3617 registros (15 %).",
        "Con las probabilidades OOF del {{modelo_elegido_nombre}}, el costo C(t) = FP(t) + 5·FN(t) se minimiza en **t = {{umbral_elegido}}** (C = {{costo_umbral}}; C/n = {{costo_n_umbral}}), con {{tp_umbral}} TP, {{fp_umbral}} FP, {{fn_umbral}} FN y {{tn_umbral}} TN: sensibilidad {{sens_umbral}} % y especificidad {{espec_umbral}} %, contactando a {{contactados_umbral}} de {{n_registros_entrenamiento}} registros ({{pct_contactados_umbral}} %).",
    ),
    (
        "Los umbrales 0,20 (1531) y 0,10 (1677) le siguen.",
        "Los umbrales {{umbral_020}} ({{costo_020}}) y {{umbral_010}} ({{costo_010}}) le siguen.",
    ),
    (
        "no contactar a nadie cuesta 5·417 = 2085, y contactar a todos 3200. El umbral 0,50 cuesta 1816 (solo detecta el 16 % de los contratantes).",
        "no contactar a nadie cuesta 5·{{n_positivos}} = {{costo_no_contactar}}, y contactar a todos {{costo_contactar_todos}}. El umbral {{umbral_050}} cuesta {{costo_050}} (solo detecta el {{sens_umbral_050}} % de los contratantes).",
    ),
    (
        "Bajar el umbral los aumenta rápidamente (917 FP con t = 0,10; 2355 con t = 0,05).",
        "Bajar el umbral los aumenta rápidamente ({{fp_010}} FP con t = {{umbral_010}}; {{fp_005}} con t = {{umbral_005}}).",
    ),
    (
        "El óptimo empírico, 0,15, es consistente con ese valor",
        "El óptimo empírico, {{umbral_elegido}}, es consistente con ese valor",
    ),
    (
        "La curva de costo es bastante plana entre 0,15 y 0,25,",
        "La curva de costo es bastante plana entre {{umbral_elegido}} y {{umbral_025}},",
    ),
]

MD_PATCHES[36] = [
    (
        "**Decisiones registradas antes de abrir la prueba:** modelo Gradient Boosting (lr = 0,05) y umbral 0,15. El modelo se reajustó con los 3617 registros",
        "**Decisiones registradas antes de abrir la prueba:** modelo {{modelo_elegido_nombre}} y umbral {{umbral_elegido}}. El modelo se reajustó con los {{n_registros_entrenamiento}} registros",
    ),
    (
        "y se evaluó una sola vez en los 904 de prueba.",
        "y se evaluó una sola vez en los {{n_test}} de prueba.",
    ),
    (
        "| Gradient Boosting | **0,3088** | **0,0886** | **0,750** |",
        "| Gradient Boosting | **{{gb_test_log_loss}}** | **{{gb_test_brier}}** | **{{gb_test_auc}}** |",
    ),
    (
        "| Referencia (p = 0,1153) | 0,3569 | 0,1018 | 0,500 |",
        "| Referencia (p = {{p_ref}}) | {{ref_test_log_loss}} | {{ref_test_brier}} | {{ref_test_auc}} |",
    ),
    (
        "Las métricas de prueba son muy parecidas a las OOF (0,311 / 0,0895 / 0,744)",
        "Las métricas de prueba son muy parecidas a las OOF ({{gb_log_loss}} / {{gb_brier}} / {{gb_auc}})",
    ),
    (
        "| GB, t = 0,15 | 55 | 112 | 49 | 688 | 52,9 % | 86,0 % | 167 | 357 | **0,395** |",
        "| GB, t = {{umbral_elegido}} | {{test_tp_015}} | {{test_fp_015}} | {{test_fn_015}} | {{test_tn_015}} | {{test_sens_015}} % | {{test_espec_015}} % | {{test_contactados_015}} | {{test_costo_015}} | **{{test_cn_015}}** |",
    ),
    (
        "| GB, t = 0,50 | 21 | 16 | 83 | 784 | 20,2 % | 98,0 % | 37 | 431 | 0,477 |",
        "| GB, t = {{umbral_050}} | {{test_tp_050}} | {{test_fp_050}} | {{test_fn_050}} | {{test_tn_050}} | {{test_sens_050}} % | {{test_espec_050}} % | {{test_contactados_050}} | {{test_costo_050}} | {{test_cn_050}} |",
    ),
    (
        "| Referencia (t = 0,15 o 0,50) | 0 | 0 | 104 | 800 | 0 % | 100 % | 0 | 520 | 0,575 |",
        "| Referencia (t = {{umbral_elegido}} o {{umbral_050}}) | 0 | 0 | {{test_n_positivos}} | {{test_n_test_menos}} | 0 % | 100 % | 0 | {{test_costo_ref}} | {{test_cn_ref}} |",
    ),
    (
        "| Contactar a todos | 104 | 800 | 0 | 0 | 100 % | 0 % | 904 | 800 | 0,885 |",
        "| Contactar a todos | {{test_n_positivos}} | {{test_n_negativos}} | 0 | 0 | 100 % | 0 % | {{n_test}} | {{test_costo_todos}} | {{test_cn_todos}} |",
    ),
    (
        "La referencia predice p = 0,115 para todos: con ambos umbrales no contacta a nadie (costo 520).",
        "La referencia predice p = {{p_ref}} para todos: con ambos umbrales no contacta a nadie (costo {{test_costo_ref}}).",
    ),
    (
        "El umbral 0,15 reduce el costo en 17 % respecto de 0,50 y en 31 % respecto de la referencia: detecta 55 de los 104 contratantes llamando solo al 18 % de la cartera, con una tasa de éxito entre los contactados de 33 % (55/167), casi el triple de la tasa base (11,5 %). En el decil de mayor probabilidad la tasa real de contratación es 44 %.",
        "El umbral {{umbral_elegido}} reduce el costo en {{test_reduccion_pct_050}} % respecto de {{umbral_050}} y en {{test_reduccion_pct_ref}} % respecto de la referencia: detecta {{test_tp_015}} de los {{test_n_positivos}} contratantes llamando solo al {{test_pct_cartera}} % de la cartera, con una tasa de éxito entre los contactados de {{test_tasa_exito}} % ({{test_tp_015}}/{{test_contactados_015}}), casi {{test_veces_tasa_base}} veces la tasa base ({{tasa_real}} %). En el decil de mayor probabilidad la tasa real de contratación es {{test_decil1_tasa}} %.",
    ),
    (
        "el modelo discrimina moderadamente (AUC 0,75)",
        "el modelo discrimina moderadamente (AUC {{gb_test_auc}})",
    ),
    (
        "y contactar a quienes tengan p ≥ 0,15 (o, si la capacidad es limitada",
        "y contactar a quienes tengan p ≥ {{umbral_elegido}} (o, si la capacidad es limitada",
    ),
    (
        "una regla con t = 0,15 es claramente mejor que no usar modelo",
        "una regla con t = {{umbral_elegido}} es claramente mejor que no usar modelo",
    ),
    (
        "con 104 positivos en prueba, la sensibilidad tiene un error estándar de ~5 puntos",
        "con {{test_n_positivos}} positivos en prueba, la sensibilidad tiene un error estándar de ~{{test_se_sens}} puntos",
    ),
]

MD_PATCHES[40] = [
    (
        "| PC1 | 1,108 | 27,7 % | 27,7 % |",
        "| PC1 | {{pca_pc1_var}} | {{pca_pc1_pct}} % | {{pca_pc1_acum}} % |",
    ),
    (
        "| PC2 | 1,067 | 26,7 % | **54,4 %** |",
        "| PC2 | {{pca_pc2_var}} | {{pca_pc2_pct}} % | **{{pca_pc2_acum}} %** |",
    ),
    (
        "| PC3 | 0,935 | 23,4 % | 77,8 % |",
        "| PC3 | {{pca_pc3_var}} | {{pca_pc3_pct}} % | {{pca_pc3_acum}} % |",
    ),
    (
        "| PC4 | 0,890 | 22,3 % | 100 % |",
        "| PC4 | {{pca_pc4_var}} | {{pca_pc4_pct}} % | 100 % |",
    ),
    (
        "**Cargas:** PC1 = 0,58·edad + 0,63·saldo − 0,38·contactos_campana + 0,35·contactos_previos → contrasta",
        "**Cargas:** PC1 = {{pca_carga_edad_pc1}}·edad {{pca_carga_saldo_pc1}}·saldo {{pca_carga_contactos_campana_pc1}}·contactos_campana {{pca_carga_contactos_previos_pc1}}·contactos_previos → contrasta",
    ),
    (
        "PC2 = −0,43·edad − 0,31·saldo − 0,58·contactos_campana + 0,62·contactos_previos → contrasta",
        "PC2 = {{pca_carga_edad_pc2}}·edad {{pca_carga_saldo_pc2}}·saldo {{pca_carga_contactos_campana_pc2}}·contactos_campana {{pca_carga_contactos_previos_pc2}}·contactos_previos → contrasta",
    ),
    (
        "(la mayor correlación es 0,097 entre edad y saldo)",
        "(la mayor correlación es {{pca_max_corr}} entre {{pca_max_corr_par}})",
    ),
    (
        "conserva solo el **54 %** de la varianza total; el 46 % restante",
        "conserva solo el **{{pca_sum_pct}} %** de la varianza total; el {{pca_omitido_pct}} % restante",
    ),
]

MD_PATCHES[43] = [
    (
        "| 2 | 11 893 | 0,301 | 2213 / 1404 |",
        "| 2 | {{k2_wss}} | {{k2_sil}} | {{k2_tamanos}} |",
    ),
    (
        "| 3 | 9 685 | 0,334 | 2090 / 1348 / 179 |",
        "| 3 | {{k3_wss}} | {{k3_sil}} | {{k3_tamanos}} |",
    ),
    (
        "| 4 | 7 665 | **0,363** | 2004 / 1289 / 177 / 147 |",
        "| 4 | {{k4_wss}} | **{{k4_sil}}** | {{k4_tamanos}} |",
    ),
    (
        "la silueta media es mayor en **k = 4**, que se elige",
        "la silueta media es mayor en **k = {{k_elegido}}**, que se elige",
    ),
    (
        "Una silueta de 0,36 indica",
        "Una silueta de {{silueta_k}} indica",
    ),
    (
        "- **Grupo 1 – Jóvenes, saldo bajo, nuevos** (n = 2004): edad 34, saldo medio 931 € (mediana 360), 2,3 contactos, casi sin campañas previas. Contratación **10,8 %**.",
        "- **Grupo 1 – Jóvenes, saldo bajo, nuevos** (n = {{g1_n}}): edad {{g1_edad}}, saldo medio {{g1_saldo}} € (mediana {{g1_saldo_med}}), {{g1_contactos}} contactos, y solo el {{g1_pct_previo}} % tiene campañas previas. Contratación **{{g1_tasa}} %**.",
    ),
    (
        "- **Grupo 2 – Mayores con más saldo** (n = 1289): edad 52, saldo medio 2146 € (mediana 624), sin historial relevante. Contratación **11,4 %**.",
        "- **Grupo 2 – Mayores con más saldo** (n = {{g2_n}}): edad {{g2_edad}}, saldo medio {{g2_saldo}} € (mediana {{g2_saldo_med}}), sin historial relevante ({{g2_pct_previo}} % con contacto previo). Contratación **{{g2_tasa}} %**.",
    ),
    (
        "- **Grupo 0 – Con historial de campañas previas** (n = 177): 100 % con contacto previo, 6,6 contactos previos en promedio. Contratación **26,6 %** (más del doble de la tasa global, 11,5 %).",
        "- **Grupo 0 – Con historial de campañas previas** (n = {{g0_n}}): {{g0_pct_previo}} % con contacto previo, {{g0_contactos_previos}} contactos previos en promedio. Contratación **{{g0_tasa}} %** (más del doble de la tasa global, {{tasa_global}} %).",
    ),
    (
        "- **Grupo 3 – Muy insistidos en esta campaña** (n = 147): 14,5 contactos en la campaña en promedio. Contratación **4,8 %**.",
        "- **Grupo 3 – Muy insistidos en esta campaña** (n = {{g3_n}}): {{g3_contactos}} contactos en la campaña en promedio. Contratación **{{g3_tasa}} %**.",
    ),
    (
        "(recordar que el plano solo muestra el 54 % de la varianza)",
        "(recordar que el plano solo muestra el {{pca_sum_pct}} % de la varianza)",
    ),
    (
        "en ninguno la contratación supera el 27 %, por lo que una regla \"grupo → clase\" predeciría siempre \"no contrata\"",
        "el grupo más favorable llega al {{k_max_tasa}} % de contratación, por lo que una regla \"grupo → clase\" predeciría siempre \"no contrata\"",
    ),
]

MD_PATCHES[47] = [
    (
        "- **Corte:** χ²₀,₉₉(4) = **13,277**.",
        "- **Corte:** χ²₀,₉₉(4) = **{{corte_chi2}}**.",
    ),
    (
        "- **Registros marcados:** **189 de 3617 (5,23 %)**.",
        "- **Registros marcados:** **{{n_marcados}} de {{n_registros_entrenamiento}} ({{pct_marcados}} %)**.",
    ),
    (
        "| 1477 | 60 | 71 188 | 1 | 0 | 518,6 | 0 |",
        "| {{anom1_id}} | {{anom1_edad}} | {{anom1_saldo}} | {{anom1_contactos_campana}} | {{anom1_contactos_previos}} | {{anom1_d2}} | {{anom1_contrata}} |",
    ),
    (
        "| 1542 | 35 | 280 | 50 | 0 | 224,3 | 0 |",
        "| {{anom2_id}} | {{anom2_edad}} | {{anom2_saldo}} | {{anom2_contactos_campana}} | {{anom2_contactos_previos}} | {{anom2_d2}} | {{anom2_contrata}} |",
    ),
]

MD_PATCHES[50] = [
    (
        "- **id 1477:** su **saldo de 71 188 €** es el máximo de la muestra, a **22,8 desviaciones estándar** de la media (la mediana es 438 € y el percentil 99 es ≈ 14 200 €). Las otras variables son normales (60 años, jubilado, 1 contacto). Como las variables están casi incorrelacionadas, la distancia se debe casi por completo a ese saldo (22,75² ≈ 518).",
        "- **id {{anom1_id}}:** su **saldo de {{anom1_saldo}} €** es el máximo de la muestra, a **{{z1_saldo}} desviaciones estándar** de la media (la mediana es {{saldo_mediana}} € y el percentil 99 es ≈ {{saldo_p99}} €). Las otras variables son normales ({{anom1_edad}} años, jubilado, {{anom1_contactos_campana}} contacto). Como las variables están casi incorrelacionadas, la distancia se debe casi por completo a ese saldo ({{z1_saldo}}² ≈ {{d2_1}}).",
    ),
    (
        "- **id 1542:** recibió **50 contactos** en la campaña, el máximo, a **14,9 desviaciones** de la media (la mediana es 2 y el percentil 99 es 16). Edad, saldo y contactos previos son típicos.",
        "- **id {{anom2_id}}:** recibió **{{anom2_contactos_campana}} contactos** en la campaña, el máximo, a **{{z2_contactos_campana}} desviaciones** de la media (la mediana es {{cc_mediana}} y el percentil 99 es {{cc_p99}}). Edad, saldo y contactos previos son típicos.",
    ),
    (
        "(asimetría: saldo 7,1, contactos_campana 4,9, contactos_previos 5,9; curtosis en exceso 99, 40 y 51)",
        "(asimetría: saldo {{skew_saldo}}, contactos_campana {{skew_contactos_campana}}, contactos_previos {{skew_contactos_previos}}; curtosis en exceso {{kurt_saldo}}, {{kurt_contactos_campana}} y {{kurt_contactos_previos}})",
    ),
    (
        "Por eso se marca el **5,2 %**, cinco veces lo esperado.",
        "Por eso se marca el **{{pct_marcados}} %**, unas {{factor_exceso}} veces el 1 % esperado.",
    ),
    (
        "y un cliente llamado 50 veces puede reflejar una política de insistencia",
        "y un cliente llamado {{anom2_contactos_campana}} veces puede reflejar una política de insistencia",
    ),
    (
        "(los marcados contratan 14,3 % vs 11,4 %)",
        "(los marcados contratan {{tasa_marcados}} % vs {{tasa_no_marcados}} %)",
    ),
]

# --------------------------------------------------------------------------
# Celda REPORTE (última celda del notebook construido)
# --------------------------------------------------------------------------
REPORTE = """# --- Celda técnica de recolección ------------------------------------------
# Publica todos los valores capturados con _pub() para poder inyectarlos en el
# texto del informe. Esta celda NO forma parte del notebook entregado.
for _clave in sorted(R):
    print(f'@@{_clave}={R[_clave]}@@')
"""


def aplicar(celdas: list[list[str]], parches: dict[int, list[tuple[str, str]]], etiqueta: str) -> None:
    fallos: list[str] = []
    for idx, reglas in sorted(parches.items()):
        src = celdas[idx][1]
        for n, (viejo, nuevo) in enumerate(reglas, start=1):
            n_oc = src.count(viejo)
            if n_oc != 1:
                fallos.append(
                    f"[{etiqueta}] celda {idx} regla {n}: el ancla aparece {n_oc} veces (se esperaba 1).\n"
                    f"--- ancla ---\n{viejo!r}"
                )
                continue
            src = src.replace(viejo, nuevo, 1)
        celdas[idx][1] = src
    if fallos:
        raise SystemExit("\n\n".join(fallos))


def literal(src: str) -> str:
    """Representa un bloque de texto como literal Python legible."""
    if "'''" not in src and not src.endswith("\\"):
        return "r'''" + src + "'''"
    return repr(src)


def main() -> None:
    nb = nbformat.read(ORIGEN, as_version=4)
    celdas: list[list[str]] = []
    for c in nb.cells:
        tipo = "md" if c.cell_type == "markdown" else "code"
        celdas.append([tipo, c.source])

    print(f"Notebook original: {len(celdas)} celdas "
          f"({sum(1 for c in celdas if c[0] == 'md')} md / {sum(1 for c in celdas if c[0] == 'code')} code)")

    aplicar(celdas, CODE_PATCHES, 'codigo')
    aplicar(celdas, MD_PATCHES, 'markdown')

    celdas.append(["code", REPORTE])

    partes = []
    for tipo, src in celdas:
        partes.append(f"    ('{tipo}', {literal(src)}),")

    cuerpo = "\n".join(partes)

    cabecera = '''# -*- coding: utf-8 -*-
"""
build_notebook_v2.py
===================

Construye de forma programatica el notebook `Grupo_36_Taller3.ipynb`
(Grupo 36 · Taller #3 "Respuesta a una campaña bancaria", Diplomado en Data
Science UC 2026).

Idea central
------------
Todo numero que aparece en el texto del informe (markdown) se escribe aqui
como un marcador `{{clave}}`. Durante la ejecucion, las celdas de codigo
publican cada valor con `_pub(clave, valor)`, que ademas lo imprime con el
formato `@@clave=valor@@`. El pipeline `run_notebook_v2.py` ejecuta el
notebook con los marcadores intactos, cosecha esos valores y vuelve a
construir el notebook reemplazando cada marcador por el valor realmente
calculado. De ese modo la prosa y las salidas no pueden quedar desalineadas.

Uso
---
    python build_notebook_v2.py            # escribe una version con marcadores
    from build_notebook_v2 import construir
    nb = construir(vals)                   # notebook final ya renderizado
"""
from __future__ import annotations

import re

import nbformat as nbf

# Lista de celdas del notebook: ('md', fuente) o ('code', fuente).
CELLS: list[tuple[str, str]] = [
'''

    pie = ''']

_PLACEHOLDER = re.compile(r"\\{\\{(\\w+)\\}\\}")


def render(tpl: str, R: dict) -> str:
    """Sustituye cada marcador `{{clave}}` por su valor en R.

    Lanza KeyError si falta algun valor: es preferible fallar de forma
    explicita a dejar un marcador sin resolver en el entregable.
    """
    def _sub(m: re.Match) -> str:
        clave = m.group(1)
        if clave not in R:
            raise KeyError(f"Placeholder sin valor: {{{{{clave}}}}}")
        return str(R[clave])

    return _PLACEHOLDER.sub(_sub, tpl)


def construir(vals: dict | None = None) -> nbf.NotebookNode:
    """Devuelve el notebook.

    vals=None  -> deja los marcadores `{{clave}}` intactos (version v0).
    vals=dict  -> renderiza los marcadores con los valores cosechados (v1).
    """
    nb = nbf.v4.new_notebook()
    celdas = []
    for tipo, src in CELLS:
        if tipo == "md":
            texto = render(src, vals) if vals is not None else src
            celdas.append(nbf.v4.new_markdown_cell(texto))
        else:
            celdas.append(nbf.v4.new_code_cell(src))
    nb.cells = celdas
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
        },
    }
    return nb


if __name__ == "__main__":
    import json
    import pathlib

    destino = pathlib.Path(__file__).resolve().parent / "Grupo_36_Taller3_v0.ipynb"
    nb = construir(None)
    nbf.write(nb, str(destino))
    print(f"Escrito {destino} ({len(nb.cells)} celdas, marcadores sin renderizar)")
    print("Marcadores presentes:", len(re.findall(r"\\{\\{\\w+\\}\\}", "\\n".join(c.source for c in nb.cells if c.cell_type == "markdown"))))
'''

    DESTINO.write_text(cabecera + cuerpo + pie, encoding="utf-8")
    print(f"Escrito {DESTINO}")

    # Comprobacion rapida: los marcadores usados existen en las celdas de codigo
    keys = set()
    for tipo, src in celdas:
        if tipo == "code":
            keys.update(re.findall(r"_pub\(\s*'([a-z0-9_]+)'", src))
            keys.update(re.findall(r"_pub\(f'([a-z0-9_]+)'", src))
    # Claves publicadas dentro de bucles (nombre construido en tiempo de ejecucion)
    dinamicas = [
        r'anom[12]_(?:id|edad|saldo|contactos_campana|contactos_previos|d2|contrata)',
        r'k[234]_(?:wss|sil|tamanos)',
        r'umbral_(?:005|010|015|020|025|050)',
        r'costo_(?:005|010|015|020|025|050)',
        r'fp_(?:005|010|015|020|025|050)',
        r'fn_(?:005|010|015|020|025|050)',
        r'pca_carga_(?:edad|saldo|contactos_campana|contactos_previos)_pc[12]',
        r'pca_pc[1234]_(?:var|pct|acum)',
        r'g[0-3]_(?:n|edad|saldo|saldo_med|contactos|contactos_previos|pct_previo|tasa)',
        r'z[12]_(?:saldo|contactos_campana)',
        r'd2_[12]',
    ]
    usados = set()
    for tipo, src in celdas:
        if tipo == "md":
            usados.update(re.findall(r"\{\{(\w+)\}\}", src))
    faltantes = set()
    for clave in usados:
        if clave in keys:
            continue
        if any(re.fullmatch(pat, clave) for pat in dinamicas):
            continue
        faltantes.add(clave)
    print(f"Marcadores usados en el texto: {len(usados)} | claves publicadas (estaticas): {len(keys)}")
    if faltantes:
        print("SIN VALOR PUBLICADO:", sorted(faltantes))
    sin_usar = sorted(keys - usados)
    if sin_usar:
        print("publicadas pero no usadas en el texto:", sin_usar)


if __name__ == "__main__":
    main()
