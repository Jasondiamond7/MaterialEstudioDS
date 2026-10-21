# -*- coding: utf-8 -*-
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
    ('md', r'''# Taller #3 · Respuesta a una campaña bancaria

Diplomado en Data Science UC 2026 · Aprendizaje Automático

Publicación: Jueves 8 de octubre de 2026, 18:30. Entrega: Miércoles 21 de octubre de 2026, 23:59.

**Grupo:** 36

**Integrantes (3 a 5):** Nadia Lell, Jaysson Leon, Kelly Martinez, Matias Reyes

**Lenguaje:** Python

**Puntaje máximo: 60. Nota = 1 + P/10.**'''),
    ('md', r'''## El caso

Un banco portugués realiza campañas telefónicas para ofrecer depósitos a plazo. Contactar a toda la cartera tiene un costo y muchos clientes no contratan el producto. El equipo comercial quiere anticipar qué registros tienen mayor probabilidad de contratación y comprender los perfiles de la campaña. Una predicción favorable que no se concreta implica un contacto poco productivo; una predicción desfavorable puede dejar fuera una oportunidad.

Como equipo de Data Science, deberán preparar los datos, comparar modelos y recomendar a qué clientes contactar. La predicción se realiza antes de la llamada registrada. El análisis de grupos y de registros inusuales complementará su recomendación.

Se utiliza la muestra pública `bank.csv` de Bank Marketing, con 4521 registros. Cada fila contiene información de un cliente y sus contactos, junto con el resultado de contratación.

## Entrega

Elijan R o Python y resuelvan todo el taller en ese mismo lenguaje. Entreguen un archivo `.zip` por grupo, llamado `Grupo_XX_Taller3.zip`, con un notebook ejecutado y los dos archivos `.csv`. Identifiquen a todos los integrantes e incluyan código, resultados e interpretación en cada respuesta. Si entregan una versión `.html`, adjunten también su fuente (`.ipynb`, `.Rmd` o `.qmd`). Usen rutas relativas e indiquen las versiones de los paquetes.'''),
    ('md', r'''## Indicaciones de trabajo

Trabajen con `clientes_entrenamiento.csv` y reserven `clientes_prueba.csv` para 3(b). Conserven las filas, fijen semilla 2026 y consideren `contrata` = 1 como clase positiva.

Usen los cinco pliegues entregados para la validación cruzada. Aprendan las transformaciones y la selección de variables solo con las filas de ajuste de cada pliegue. Reúnan predicciones fuera de pliegue (OOF): probabilidades para los modelos probabilísticos y puntajes de decisión para SVM.

El modelo de referencia predice la proporción positiva de las filas de ajuste. Para la evaluación final, calculen esa proporción con todo el conjunto de entrenamiento.'''),
    ('md', r'''## Datos

Entrenamiento: 3617 registros. Prueba: 904 registros.

| Columna | Descripción |
|---|---|
| `edad` / `saldo` | Edad en años / saldo medio anual en euros. |
| `ocupacion` / `estado_civil` / `educacion` | Categorías de ocupación, estado civil y educación. |
| `mora` / `hipoteca` / `prestamo` | Indicadores yes/no de mora, crédito hipotecario y préstamo personal. |
| `canal` / `dia` / `mes` | Medio de contacto y día del mes / mes de la llamada registrada. |
| `duracion` | Duración de la llamada registrada, en segundos. |
| `contactos_campana` | Número de contactos de la campaña, incluida la llamada registrada. |
| `dias_previos` | Días desde el contacto de una campaña anterior; -1 indica que no lo hubo. |
| `contactos_previos` / `resultado_previo` | Número de contactos y resultado de campañas anteriores. |
| `contrata` | Respuesta: 1 = contrata el depósito; 0 = no contrata. |
| `id_cliente` / `pliegue` | Identificador del registro / pliegue de validación (solo en entrenamiento). |

En las preguntas 4 y 5 usen solo `edad`, `saldo`, `contactos_campana` y `contactos_previos` del conjunto de entrenamiento, estandarizadas con media y desviación muestral.

Fuente: [Bank Marketing (UCI)](https://doi.org/10.24432/C5K306). Moro, Rita y Cortez (2014). Licencia CC BY 4.0. Muestra completa, con nombres traducidos y particiones preparadas para el taller.'''),
    ('md', r'''## Preparación

Funciones de apoyo: OneHotEncoder, StandardScaler, f_classif/SelectKBest, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, SVC y decision_function, GaussianNB, PCA y KMeans.'''),
    ('md', r'''**Google Colab:** ejecuten la siguiente celda para subir los dos `.csv` (o súbanlos al panel *Archivos* de la izquierda). Luego usen *Entorno de ejecución → Ejecutar todas*.'''),
    ('code', r'''# --- Configuración para Google Colab -------------------------------------
# Si los CSV no están en la carpeta de trabajo, Colab pedirá subirlos.
from pathlib import Path
import os, sys, warnings
try:
    import google.colab  # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False
necesarios = ['clientes_entrenamiento.csv', 'clientes_prueba.csv']
if EN_COLAB and not all(Path(f).exists() for f in necesarios):
    from google.colab import files
    print('Suban clientes_entrenamiento.csv y clientes_prueba.csv')
    subidos = files.upload()
    # Si el navegador agregó sufijos, renombrar al nombre esperado
    for nombre in subidos:
        for f in necesarios:
            if f.split('.')[0] in nombre and nombre != f:
                os.rename(nombre, f)
print('Archivos presentes:', [f for f in necesarios if Path(f).exists()])'''),
    ('code', r'''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import sklearn, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (log_loss, brier_score_loss, roc_auc_score, roc_curve,
                             confusion_matrix, silhouette_score)
warnings.filterwarnings('ignore', category=UserWarning)
pd.set_option('display.width', 140); pd.set_option('display.max_columns', 30)
SEMILLA = 2026
np.random.seed(SEMILLA)

R = {}   # valores que alimentan el texto del informe


def _pub(clave, valor):
    """Registra un valor para el texto y lo publica en formato parseable."""
    R[clave] = valor
    print(f'@@{clave}={valor}@@')


def _num(x, dec=4):
    """Formatea un número al estilo español (coma decimal)."""
    return f'{float(x):.{dec}f}'.replace('.', ',')


CARPETA = Path('.')
entrenamiento = pd.read_csv(CARPETA / 'clientes_entrenamiento.csv')
print('Python:', sys.version.split()[0], '| numpy:', np.__version__, '| pandas:', pd.__version__,
      '| sklearn:', sklearn.__version__, '| scipy:', scipy.__version__, '| matplotlib:', matplotlib.__version__)
print(entrenamiento.shape)
# El conjunto de prueba se abre en 3(b).'''),
    ('md', r'''## Pregunta 1. Problema, preparación y representación (12 puntos)'''),
    ('md', r'''### 1(a) (3 puntos)

Identifiquen la tarea de aprendizaje, la unidad de observación y la respuesta. ¿Por qué `duracion` no puede usarse para predecir antes de la llamada? Exclúyanla de los predictores, junto con `id_cliente`, `pliegue` y `contrata`.'''),
    ('code', r'''df = pd.read_csv('clientes_entrenamiento.csv')
print(df.shape)
df.head()'''),
    ('code', r'''RESPUESTA = 'contrata'
EXCLUIDAS = ['duracion', 'id_cliente', 'pliegue', RESPUESTA]
predictores_originales = [c for c in entrenamiento.columns if c not in EXCLUIDAS]
print('Predictores originales (%d):' % len(predictores_originales), predictores_originales)
print('Respuesta:', RESPUESTA, '-> valores', sorted(entrenamiento[RESPUESTA].unique()))
# duracion se conoce solo al terminar la llamada; su relación con la respuesta lo muestra:
print(entrenamiento.groupby(RESPUESTA)['duracion'].describe()[['mean', '50%']])

# Valores que el texto de la respuesta cita (se publican para poder inyectarlos)
_pub('n_registros_entrenamiento', len(entrenamiento))
_pub('n_predictores_originales', len(predictores_originales))
_pub('duracion_media_contrata', _num(entrenamiento.loc[entrenamiento[RESPUESTA] == 1, 'duracion'].mean(), 0))
_pub('duracion_media_no_contrata', _num(entrenamiento.loc[entrenamiento[RESPUESTA] == 0, 'duracion'].mean(), 0))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

- **Tarea:** aprendizaje supervisado de **clasificación binaria** (y, además, estimación de una probabilidad de contratación para ordenar y priorizar clientes).
- **Unidad de observación:** cada fila es un registro cliente–contacto de la campaña (un cliente con su llamada registrada); hay {{n_registros_entrenamiento}} en entrenamiento.
- **Respuesta:** `contrata` (1 = contrata el depósito a plazo, clase positiva; 0 = no contrata).

**¿Por qué no `duracion`?** La duración de la llamada registrada solo se conoce **después** de que la llamada termina, y en ese momento ya se sabe en gran medida si el cliente contrata (una llamada larga suele ser una venta: la duración media es {{duracion_media_contrata}} s en quienes contratan vs {{duracion_media_no_contrata}} s en quienes no). Usarla sería **fuga de información** (*data leakage*): el modelo se vería muy bueno en validación, pero sería inaplicable para decidir *a quién llamar*, que es justamente el uso previsto. Por eso se excluye junto con `id_cliente` (identificador sin contenido predictivo), `pliegue` (artefacto del diseño de validación) y `contrata` (la respuesta). Quedan {{n_predictores_originales}} predictores originales.'''),
    ('md', r'''### 1(b) (4 puntos)

Revisen tipos de variables, faltantes, categorías unknown, duplicados y balance de clases. Interpreten los valores negativos de `saldo` y el código -1 de `dias_previos`. ¿Sería suficiente la exactitud de predecir siempre la clase mayoritaria? Fundamenten.'''),
    ('code', r'''print('Tipos de variables:'); print(entrenamiento.dtypes.astype(str).value_counts())
categoricas = entrenamiento[predictores_originales].select_dtypes(exclude='number').columns.tolist()
numericas = entrenamiento[predictores_originales].select_dtypes(include='number').columns.tolist()
print('Categóricas:', categoricas); print('Numéricas:', numericas)
print('\nFaltantes (NA) totales:', int(entrenamiento.isna().sum().sum()))
print('\nCategorías "unknown" por variable:')
print(pd.Series({c: int((entrenamiento[c] == 'unknown').sum()) for c in categoricas}).to_frame('n_unknown')
      .assign(pct=lambda t: (100 * t.n_unknown / len(entrenamiento)).round(1)))
print('\nFilas duplicadas (todas las columnas):', int(entrenamiento.duplicated().sum()),
      '| duplicadas sin id_cliente/pliegue:', int(entrenamiento.drop(columns=['id_cliente', 'pliegue']).duplicated().sum()),
      '| id_cliente repetidos:', int(entrenamiento.id_cliente.duplicated().sum()))
print('\nBalance de clases:'); print(entrenamiento[RESPUESTA].value_counts().to_frame('n')
      .assign(prop=lambda t: (t.n / t.n.sum()).round(4)))
print('\nSaldo negativo:', int((entrenamiento.saldo < 0).sum()), 'registros;',
      'mínimo', entrenamiento.saldo.min(), '| saldo = 0:', int((entrenamiento.saldo == 0).sum()))
print('dias_previos = -1:', int((entrenamiento.dias_previos == -1).sum()),
      '| coincide con contactos_previos = 0:',
      bool(((entrenamiento.dias_previos == -1) == (entrenamiento.contactos_previos == 0)).all()))
print(pd.crosstab(entrenamiento.dias_previos == -1, entrenamiento.resultado_previo))
acc_mayoritaria = (entrenamiento[RESPUESTA] == 0).mean()
print(f'\nExactitud de predecir siempre "no contrata": {acc_mayoritaria:.4f} (sensibilidad = 0)')
print('Balance por pliegue:'); print(entrenamiento.groupby('pliegue')[RESPUESTA].agg(['size', 'mean']).round(4))

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
_pub('acc_mayoritaria', _num(100 * acc_mayoritaria, 1))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

- **Tipos:** {{n_numericas}} numéricas enteras (`edad`, `saldo`, `dia`, `contactos_campana`, `dias_previos`, `contactos_previos`) y {{n_categoricas}} categóricas (`ocupacion`, `estado_civil`, `educacion`, `mora`, `hipoteca`, `prestamo`, `canal`, `mes`, `resultado_previo`). `dia` se trata como numérica y `mes` como categórica.
- **Faltantes:** no hay NA explícitos, pero sí categorías **unknown**: `resultado_previo` {{unk_resultado_previo_pct}} % (no es un faltante real: corresponde exactamente a quienes no tuvieron campaña previa), `canal` {{unk_canal_pct}} %, `educacion` {{unk_educacion_pct}} % y `ocupacion` {{unk_ocupacion_pct}} %. Como unknown puede ser informativo (p. ej. canal no registrado), se conserva como categoría.
- **Duplicados:** {{n_duplicados}} filas duplicadas (ni considerando solo las variables de contenido) y ningún `id_cliente` repetido.
- **Balance:** {{n_clase0}} no contratan ({{prop_clase0}} %) y {{n_clase1}} contratan ({{prop_clase1}} %); la proporción es casi idéntica en los 5 pliegues (estratificados).
- **`saldo` negativo** ({{n_saldo_neg}} registros, mínimo {{min_saldo}} €): no es un error; representa clientes **sobregirados** o con saldo medio deudor en el año. Es una situación financiera relevante que se conserva.
- **`dias_previos` = −1** ({{n_dias_m1}} registros): es un **código** que significa "no fue contactado en una campaña anterior", no una cantidad de días. Coincide exactamente con `contactos_previos` = 0 y con `resultado_previo` = unknown. Tratarlo como número (−1 días) sería incoherente, por lo que en 1(c) se separa en un indicador.

**¿Basta la exactitud de la clase mayoritaria?** No. Predecir siempre "no contrata" logra una exactitud de **{{acc_mayoritaria}} %**, pero tiene sensibilidad 0: no identifica a ningún cliente que contrataría, que es precisamente lo que interesa al banco. Con clases desbalanceadas la exactitud es engañosa; hay que usar métricas que evalúen la calidad de las probabilidades (log loss, Brier), el ordenamiento (AUC) y los costos asimétricos de FP y FN (pregunta 3).'''),
    ('md', r'''### 1(c) (5 puntos)

Construyan dos variables: `contactado_antes` = 1 si `dias_previos` es distinto de -1, y `log_contactos_previos` = log(1 + `contactos_previos`). Expliquen su sentido. Reemplacen -1 por 0 en `dias_previos` conservando el indicador, mantengan unknown como categoría y codifiquen las variables categóricas mediante one-hot. Describan y apliquen el escalamiento necesario para cada método.'''),
    ('code', r'''def construir_variables(df):
    """Transformaciones fila a fila (no aprenden nada de los datos: no hay fuga)."""
    d = df.copy()
    d['contactado_antes'] = (d['dias_previos'] != -1).astype(int)
    d['log_contactos_previos'] = np.log1p(d['contactos_previos'])
    d['dias_previos'] = d['dias_previos'].replace(-1, 0)
    return d

NUM = ['edad', 'saldo', 'dia', 'contactos_campana', 'dias_previos', 'log_contactos_previos']
BIN = ['contactado_antes']
CAT = ['ocupacion', 'estado_civil', 'educacion', 'mora', 'hipoteca', 'prestamo',
       'canal', 'mes', 'resultado_previo']

def nuevo_preparador():
    """Aprende medias/desv. (numéricas) y categorías (one-hot) SOLO con las filas que recibe en fit.
    'unknown' se mantiene como categoría propia. drop='first' deja una categoría de referencia,
    evitando la colinealidad perfecta con el intercepto en la regresión logística sin penalización.
    """
    return ColumnTransformer([
        ('num', StandardScaler(), NUM),
        ('bin', 'passthrough', BIN),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CAT),
    ], verbose_feature_names_out=False)

datos = construir_variables(entrenamiento)
print(datos[['dias_previos', 'contactado_antes', 'contactos_previos', 'log_contactos_previos']].describe().round(3))
print('Asimetría contactos_previos: %.2f -> log: %.2f' % (entrenamiento.contactos_previos.skew(), datos.log_contactos_previos.skew()))
print('Tasa de contratación según contactado_antes:'); print(datos.groupby('contactado_antes')[RESPUESTA].agg(['size', 'mean']).round(3))

# Ejemplo de ajuste en todo el entrenamiento (solo para describir la matriz; en CV se ajusta por pliegue)
prep_demo = nuevo_preparador().fit(datos)
X_demo = prep_demo.transform(datos)
COLUMNAS = list(prep_demo.get_feature_names_out())
print('\nNúmero de columnas preparadas p =', len(COLUMNAS))
print(COLUMNAS)
print('\nNuméricas estandarizadas (media ~0, desv ~1):')
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
_pub('saldo_desv', _num(entrenamiento.saldo.std(ddof=1), 0))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**Variables construidas:**
- `contactado_antes` = 1 si `dias_previos` ≠ −1: indica si el cliente participó en una campaña anterior. Separa el "código" del valor numérico, de modo que `dias_previos` (con −1 → 0) mide la recencia solo cuando tiene sentido y el indicador capta el salto cualitativo. Es informativo: la tasa de contratación es {{tasa_contactado_si}} % entre los contactados antes vs {{tasa_contactado_no}} % en el resto.
- `log_contactos_previos` = log(1 + `contactos_previos`): la variable original es muy asimétrica (asimetría {{skew_contactos_previos}}, máximo {{max_contactos_previos}}); el logaritmo comprime la cola (asimetría {{skew_log_contactos}}), reduce la influencia de valores extremos y refleja rendimientos decrecientes (pasar de 0 a 1 contacto importa más que de 10 a 11). El +1 permite el valor 0. Usamos la versión logarítmica en lugar de la original para no repetir la misma información.

**Preparación** (función `nuevo_preparador`, que en la validación cruzada se ajusta **solo con las filas de ajuste de cada pliegue**, para evitar fuga):
- `dias_previos`: −1 reemplazado por 0, conservando `contactado_antes`.
- Categóricas (incluida unknown como categoría) → **one-hot** con una categoría de referencia (`drop='first'`), lo que evita la colinealidad perfecta con el intercepto en la regresión logística sin penalización. `handle_unknown='ignore'` protege ante categorías no vistas en un pliegue.
- Resultado: **p = {{p_columnas}} columnas** ({{n_num_preparadas}} numéricas, {{n_bin_preparadas}} indicador y {{n_dummies}} dummies).

**Escalamiento según método:**
- **Regresión logística:** no cambia las predicciones de un modelo sin penalización, pero estandarizar (media 0, desv. 1) mejora la convergencia numérica y hace comparables los coeficientes.
- **SVM radial:** **imprescindible**: el kernel RBF depende de distancias euclidianas, y sin escalar `saldo` (desv. ≈ {{saldo_desv}}) dominaría completamente la distancia; además gamma = 1/p presupone columnas de escala comparable.
- **Selección F de ANOVA, PCA, K-means y Mahalanobis:** sensibles a la escala (las tres últimas se trabajan con variables estandarizadas, como pide el enunciado).
- **Árboles, Random Forest y boosting:** invariantes a transformaciones monótonas; no requieren escalamiento (se les entrega la misma matriz, sin efecto en sus particiones).
- **Naive Bayes gaussiano:** estima una media y varianza por columna y clase, por lo que es invariante a reescalamientos lineales; no lo necesita.

Por simplicidad se usa una matriz común: numéricas estandarizadas y dummies 0/1 sin escalar (las dummies ya están en una escala acotada).'''),
    ('md', r'''## Pregunta 2. Selección de variables y modelos supervisados (24 puntos)'''),
    ('md', r'''| Modelo | Configuraciones |
|---|---|
| Regresión logística | Sin penalización: todas las columnas preparadas y una versión con selección de 10 columnas. |
| Árbol de decisión | Profundidades máximas 3 y 6; mínimo 20 observaciones por hoja. |
| Random Forest | 200 árboles; raíz cuadrada del número de columnas como candidatas por división; mínimo 10 por hoja. |
| Boosting | Una variante vista en clases; hasta 200 árboles y profundidad máxima 3. Informen variante y tasa de aprendizaje. |
| SVM radial | C = 1 y gamma = 1/p, donde p es el número de columnas preparadas; usen el puntaje de decisión. |
| Naive Bayes | Variante gaussiana sobre las columnas numéricas y binarias preparadas. |'''),
    ('md', r'''### 2(a) (5 puntos)

Comparen regresión logística con todas las columnas preparadas y con las 10 de mayor estadístico F de ANOVA respecto de `contrata`. Informen las columnas seleccionadas en cada pliegue y comparen sus métricas OOF. ¿En qué se diferencia esta selección de PCA? ¿Qué limitación tiene el filtro univariado?'''),
    ('code', r'''y = datos[RESPUESTA].to_numpy()
pliegues = datos['pliegue'].to_numpy()
EPS = 1e-15

def metricas(y_true, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {'log_loss': log_loss(y_true, p, labels=[0, 1]),
            'brier': brier_score_loss(y_true, p),
            'auc': roc_auc_score(y_true, p)}

def validar(crear_modelo, k_sel=None, puntaje=False, registrar_sel=None):
    """CV con los 5 pliegues entregados. Preparación (y selección) aprendidas solo en las filas de ajuste."""
    oof = np.zeros(len(datos)); ll_entr = []
    for f in sorted(np.unique(pliegues)):
        aj, va = pliegues != f, pliegues == f
        prep = nuevo_preparador().fit(datos[aj])
        Xa, Xv = prep.transform(datos[aj]), prep.transform(datos[va])
        nombres = prep.get_feature_names_out()
        if k_sel is not None:
            sel = SelectKBest(f_classif, k=k_sel).fit(Xa, y[aj])
            Xa, Xv = sel.transform(Xa), sel.transform(Xv)
            if registrar_sel is not None:
                registrar_sel[f] = list(nombres[sel.get_support()])
        m = crear_modelo(Xa.shape[1]).fit(Xa, y[aj])
        if puntaje:
            oof[va] = m.decision_function(Xv)          # positivo -> clase 1
        else:
            oof[va] = m.predict_proba(Xv)[:, 1]
            ll_entr.append(log_loss(y[aj], np.clip(m.predict_proba(Xa)[:, 1], EPS, 1 - EPS)))
    return oof, ll_entr

def logistica_sin_penalizacion():
    # sklearn >= 1.8 depreca penalty=None y usa C=np.inf; versiones anteriores (p. ej. Colab) usan penalty=None
    version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    if version >= (1, 8):
        return LogisticRegression(C=np.inf, max_iter=10000)
    return LogisticRegression(penalty=None, max_iter=10000)
logit = lambda p: logistica_sin_penalizacion()
seleccion = {}
oof = {}; ll_train = {}
oof['Logística (todas)'], ll_train['Logística (todas)'] = validar(logit)
oof['Logística (10 F-ANOVA)'], ll_train['Logística (10 F-ANOVA)'] = validar(logit, k_sel=10, registrar_sel=seleccion)

tabla_sel = pd.DataFrame({f'Pliegue {f}': sorted(c) for f, c in seleccion.items()})
print('Columnas seleccionadas por pliegue (orden alfabético):'); print(tabla_sel.to_string())
frec = pd.Series([c for cs in seleccion.values() for c in cs]).value_counts()
print('\nFrecuencia de selección en los 5 pliegues:'); print(frec.to_string())
print()
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
_pub('sel_pliegue_estado_civil', int(max(f for f, c in seleccion.items() if 'estado_civil_married' in c)))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**Selección por pliegue:** el filtro elige casi las mismas 10 columnas en los 5 pliegues: `dias_previos`, `log_contactos_previos`, `contactado_antes`, `resultado_previo_success`, `resultado_previo_unknown`, `canal_unknown`, `hipoteca_yes`, `mes_may`, `mes_oct` (las {{sel_n_todas}} anteriores aparecen en los 5 pliegues) y `mes_mar` ({{sel_mes_mar_folds}}/5); en el pliegue {{sel_pliegue_estado_civil}} entra `estado_civil_married` en lugar de `mes_mar`. La selección es **estable**, pero muy **redundante**: varias de las columnas seleccionadas (`dias_previos`, `log_contactos_previos`, `contactado_antes`, `resultado_previo_unknown`) codifican esencialmente lo mismo (si hubo campaña previa).

**Métricas OOF:**

| Modelo | Log loss | Brier | AUC |
|---|---|---|---|
| Logística, 42 columnas | **{{log_todas_log_loss}}** | **{{log_todas_brier}}** | **{{log_todas_auc}}** |
| Logística, 10 columnas F | {{log10_log_loss}} | {{log10_brier}} | {{log10_auc}} |

La versión con todas las columnas es mejor en las tres métricas: la selección pierde información útil (p. ej. otros meses, edad, `contactos_campana`, `prestamo`), que con {{n_registros_entrenamiento}} observaciones el modelo completo puede estimar sin sobreajustar en exceso. La selección gana en simplicidad e interpretabilidad, a costo de ~{{delta_auc_logistica}} de AUC.

**Diferencia con PCA:** la selección F **conserva columnas originales** (interpretables) y es **supervisada** (usa `contrata` para ordenar). PCA **construye combinaciones lineales nuevas** de todas las columnas y es **no supervisado**: maximiza varianza explicada sin mirar la respuesta, así que las componentes de mayor varianza no necesariamente son las más predictivas, y no elimina la necesidad de medir todas las variables.

**Limitación del filtro univariado:** evalúa cada columna **por separado** (diferencia de medias entre clases). Por eso (i) **no detecta redundancia**: selecciona varias columnas casi colineales, como se observa; (ii) **ignora interacciones y efectos conjuntos**: una variable débil por sí sola pero útil en combinación con otras queda fuera; (iii) solo capta relaciones lineales en la media. Además, k = 10 es arbitrario y no se eligió optimizando el desempeño.'''),
    ('md', r'''### 2(b) (11 puntos)

Comparen las configuraciones de la tabla, incluidas las dos regresiones de 2(a), mediante sus predicciones OOF. Presenten log loss, Brier y AUC; para SVM, solo AUC. Incluyan el modelo de referencia y, para ambos árboles, el log loss medio de entrenamiento de los cinco ajustes. Seleccionen el modelo probabilístico con menor log loss OOF.'''),
    ('code', r'''modelos = {
    'Árbol (prof. 3)': (lambda p: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEMILLA), False),
    'Árbol (prof. 6)': (lambda p: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEMILLA), False),
    'Random Forest': (lambda p: RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=10,
                                                       random_state=SEMILLA, n_jobs=-1), False),
    'Gradient Boosting (lr=0.05)': (lambda p: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                                          random_state=SEMILLA), False),
    'Naive Bayes gaussiano': (lambda p: GaussianNB(), False),
    'SVM radial (C=1, γ=1/p)': (lambda p: SVC(kernel='rbf', C=1.0, gamma=1.0 / p), True),
}
for nombre, (crear, es_puntaje) in modelos.items():
    oof[nombre], ll_train[nombre] = validar(crear, puntaje=es_puntaje)

# Modelo de referencia: proporción positiva de las filas de ajuste de cada pliegue
oof_ref = np.zeros(len(datos))
for f in np.unique(pliegues):
    oof_ref[pliegues == f] = y[pliegues != f].mean()
oof['Referencia (proporción)'] = oof_ref

filas = []
for nombre, s in oof.items():
    if nombre.startswith('SVM'):
        filas.append({'modelo': nombre, 'log_loss': np.nan, 'brier': np.nan, 'auc': roc_auc_score(y, s)})
    else:
        r = metricas(y, s); r['modelo'] = nombre
        r['log_loss_entren_medio'] = np.mean(ll_train[nombre]) if nombre.startswith('Árbol') else np.nan
        filas.append(r)
resultados = pd.DataFrame(filas).set_index('modelo')[['log_loss', 'brier', 'auc', 'log_loss_entren_medio']]
resultados = resultados.sort_values('log_loss')
print(resultados.round(4).to_string())
print('\nNota: con referencia el AUC es 0.5 porque la probabilidad es casi constante; SVM solo informa AUC.')

MODELO_ELEGIDO = resultados['log_loss'].idxmin()
print('\nModelo probabilístico con menor log loss OOF:', MODELO_ELEGIDO)

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
_pub('mejora_pct_gb', _num(100 * (resultados.loc['Referencia (proporción)', 'log_loss'] - resultados.loc['Gradient Boosting (lr=0.05)', 'log_loss']) / resultados.loc['Referencia (proporción)', 'log_loss'], 0))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**Tabla OOF (5 pliegues entregados), ordenada por log loss:**

| Modelo | Log loss | Brier | AUC | Log loss entren. (medio) |
|---|---|---|---|---|
| **Gradient Boosting** (lr = 0,05; 200 árboles; prof. 3) | **{{gb_log_loss}}** | **{{gb_brier}}** | **{{gb_auc}}** | – |
| Random Forest | {{rf_log_loss}} | {{rf_brier}} | {{rf_auc}} | – |
| Logística (42 columnas) | {{log_todas_log_loss}} | {{log_todas_brier}} | {{log_todas_auc}} | – |
| Logística (10 F) | {{log10_log_loss}} | {{log10_brier}} | {{log10_auc}} | – |
| Árbol prof. 3 | {{arb3_log_loss}} | {{arb3_brier}} | {{arb3_auc}} | {{arb3_log_loss_train}} |
| Árbol prof. 6 | {{arb6_log_loss}} | {{arb6_brier}} | {{arb6_auc}} | {{arb6_log_loss_train}} |
| Referencia (proporción) | {{ref_log_loss}} | {{ref_brier}} | {{ref_auc}} | – |
| Naive Bayes gaussiano | {{nb_log_loss}} | {{nb_brier}} | {{nb_auc}} | – |
| SVM radial (C = 1, γ = 1/{{p_columnas}}) | – | – | {{svm_auc}} | – |

**Boosting:** usamos **Gradient Boosting** (boosting por gradiente con pérdida log loss/desviación binomial), 200 árboles de profundidad máxima 3 y **tasa de aprendizaje 0,05**.

**Modelo seleccionado: {{modelo_elegido}}**, con el menor log loss OOF ({{gb_log_loss}}). También tiene el mejor AUC y un Brier prácticamente igual al de la logística ({{gb_brier}} frente a {{log_todas_brier}}). Las diferencias con Random Forest y la logística son pequeñas (≈ {{delta_ll_gb_rf}}–{{delta_ll_gb_log}} en log loss), por lo que los tres son razonables; el criterio preestablecido define el ganador.

Todos los modelos (salvo Naive Bayes) superan a la referencia, pero la mejora en log loss es moderada ({{ref_log_loss}} → {{gb_log_loss}}, ≈ {{mejora_pct_gb}} %): sin `duracion`, la información disponible antes de la llamada discrimina solo de manera moderada (AUC ≈ {{gb_auc}}). Naive Bayes es **peor que la referencia** en log loss y Brier pese a un AUC aceptable (ver 2(c)). Para los árboles, el log loss se evaluó con probabilidades recortadas a [10⁻¹⁵, 1 − 10⁻¹⁵].'''),
    ('md', r'''### 2(c) (8 puntos)

Interpreten los resultados considerando: profundidad y sobreajuste de los árboles; diferencias entre bagging, Random Forest y boosting; independencia condicional de Naive Bayes y limitaciones de la variante gaussiana para columnas binarias; efecto de C, gamma y el escalamiento en SVM. ¿Un mejor ordenamiento de las observaciones implica mejores probabilidades? Fundamenten.'''),
    ('code', r'''# Evidencia de apoyo: sobreajuste de árboles, calibración y relación ordenamiento vs. probabilidades
arb = resultados.loc[['Árbol (prof. 3)', 'Árbol (prof. 6)'], ['log_loss_entren_medio', 'log_loss']]
arb['brecha'] = arb['log_loss'] - arb['log_loss_entren_medio']
print('Árboles: log loss entrenamiento vs OOF'); print(arb.round(4))

print('\nRango de probabilidades OOF de Naive Bayes:',
      'proporción < 0.01 = %.3f, > 0.99 = %.3f' % ((oof['Naive Bayes gaussiano'] < .01).mean(), (oof['Naive Bayes gaussiano'] > .99).mean()))
print('Probabilidad media OOF por modelo vs tasa real (%.4f):' % y.mean())
print(pd.Series({k: v.mean() for k, v in oof.items() if not k.startswith('SVM')}).round(4).to_string())

# Curvas de calibración (deciles de probabilidad OOF)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for nombre in [MODELO_ELEGIDO, 'Logística (todas)', 'Random Forest', 'Naive Bayes gaussiano', 'Árbol (prof. 6)']:
    p = oof[nombre]
    q = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False)
    cal = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').mean()
    ax[0].plot(cal.p, cal.y, marker='o', label=nombre)
ax[0].plot([0, 1], [0, 1], 'k--', lw=1); ax[0].set_xlim(0, .8); ax[0].set_ylim(0, .8)
ax[0].set_xlabel('Probabilidad media predicha (decil)'); ax[0].set_ylabel('Tasa observada'); ax[0].set_title('Calibración OOF'); ax[0].legend(fontsize=8)
for nombre, s in oof.items():
    if nombre.startswith('Referencia'):
        continue
    fpr, tpr, _ = roc_curve(y, s); ax[1].plot(fpr, tpr, lw=1.2, label=f'{nombre} ({roc_auc_score(y, s):.3f})')
ax[1].plot([0, 1], [0, 1], 'k--', lw=1); ax[1].set_xlabel('1 - especificidad'); ax[1].set_ylabel('Sensibilidad')
ax[1].set_title('Curvas ROC OOF'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()

# Efecto de C y gamma en SVM (AUC OOF) y del escalamiento
p_cols = len(COLUMNAS)
grilla = []
for Cval in [0.1, 1, 10]:
    for g_mult in [0.1, 1, 10]:
        s, _ = validar(lambda p, Cval=Cval, g_mult=g_mult: SVC(kernel='rbf', C=Cval, gamma=g_mult / p), puntaje=True)
        grilla.append({'C': Cval, 'gamma': f'{g_mult}/p', 'auc': roc_auc_score(y, s)})
print('\nSVM: AUC OOF para distintos C y gamma'); print(pd.DataFrame(grilla).pivot(index='C', columns='gamma', values='auc').round(4))

# SVM sin estandarizar las numéricas (escala original)
def validar_svm_sin_escala():
    s = np.zeros(len(datos))
    for f in np.unique(pliegues):
        aj, va = pliegues != f, pliegues == f
        prep = ColumnTransformer([('num', 'passthrough', NUM), ('bin', 'passthrough', BIN),
                                  ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CAT)]).fit(datos[aj])
        Xa, Xv = prep.transform(datos[aj]), prep.transform(datos[va])
        s[va] = SVC(kernel='rbf', C=1, gamma=1 / Xa.shape[1]).fit(Xa, y[aj]).decision_function(Xv)
    return s
auc_sin_escala = roc_auc_score(y, validar_svm_sin_escala())
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
_pub('n_candidatas_rf', int(np.sqrt(len(COLUMNAS))))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**Profundidad y sobreajuste de árboles.** El árbol de profundidad 6 tiene menor log loss de entrenamiento ({{arb6_log_loss_train}} vs {{arb3_log_loss_train}}) pero **peor** log loss OOF ({{arb6_log_loss}} vs {{arb3_log_loss}}): la brecha entrenamiento–validación crece de {{arb3_brecha}} a {{arb6_brecha}}. Más profundidad reduce sesgo pero aumenta varianza: las hojas más pequeñas producen probabilidades extremas y poco estables que se castigan en log loss. Aun así, el árbol 6 ordena algo mejor (AUC {{arb6_auc}} vs {{arb3_auc}}) y tiene menor Brier: un árbol de 3 niveles solo genera ≤ 8 probabilidades distintas, muy poca resolución. Ningún árbol individual se acerca a los ensambles.

**Bagging, Random Forest y boosting.** *Bagging* promedia árboles profundos ajustados a muestras bootstrap: reduce la **varianza** (no el sesgo), aunque los árboles quedan correlacionados porque todos eligen las mismas variables fuertes (`resultado_previo`, `mes`, contacto previo). *Random Forest* añade la selección aleatoria de √p columnas candidatas por división (con p = {{p_columnas}}, son {{n_candidatas_rf}} candidatas), lo que **decorrelaciona** los árboles y hace más efectivo el promedio; min. 10 por hoja suaviza probabilidades. *Boosting* es **secuencial**: cada árbol pequeño (prof. 3) se ajusta a los residuos/gradientes del ensamble actual, con una tasa de aprendizaje que contrae cada aporte; reduce principalmente el **sesgo**, y su riesgo de sobreajuste se controla con la tasa, el número de árboles y la profundidad. Aquí el boosting obtiene el mejor log loss y AUC, y RF queda muy cerca: ambos capturan no linealidades e interacciones (p. ej. mes × canal) que la logística no modela.

**Naive Bayes.** Supone **independencia condicional** de las columnas dado `contrata`, falso aquí: `dias_previos`, `contactado_antes`, `log_contactos_previos` y `resultado_previo_unknown` son casi la misma información, y las dummies de una misma variable son mutuamente excluyentes. Multiplicar verosimilitudes de evidencia redundante la "cuenta varias veces" y empuja las probabilidades a los extremos: el {{nb_frac_bajo001}} % de las probabilidades OOF son < 0,01 y el {{nb_frac_sobre099}} % > 0,99, y la media es {{nb_media}} (tasa real {{tasa_real}} %). Además, la variante gaussiana modela columnas **binarias** (0/1) como normales, lo que no corresponde: con dummies raras la varianza estimada es minúscula y la densidad diverge, generando razones de verosimilitud enormes. Resultado: **mal calibrado** (log loss {{nb_log_loss}}, peor que la referencia) aunque ordena de forma razonable (AUC {{nb_auc}}). Una variante Bernoulli/categórica para las binarias sería más apropiada.

**SVM: C, gamma y escalamiento.** C controla el costo de violar el margen: C grande → menos regularización, frontera más ajustada (más varianza); C pequeño → margen más amplio. Gamma controla el alcance del kernel RBF: gamma grande → influencia muy local, fronteras irregulares (sobreajuste); gamma pequeño → fronteras casi lineales. En la grilla (AUC OOF), γ = 0,1/p da los peores resultados ({{svm_auc_g01_min}}–{{svm_auc_g01_max}}) y γ = 1/p o 10/p mejoran ({{svm_auc_g1_min}}–{{svm_auc_g1_max}} y {{svm_auc_g10_min}}–{{svm_auc_g10_max}}), con poco efecto de C. El **escalamiento** es crítico: sin estandarizar las numéricas el AUC cae de {{svm_auc}} a **{{svm_auc_sin_escala}}**, porque `saldo` domina las distancias. La SVM no entrega probabilidades sino un puntaje de decisión (distancia con signo a la frontera); por eso solo se evalúa con AUC, y para usarla en costos habría que calibrarla (Platt o isotónica).

**¿Mejor ordenamiento implica mejores probabilidades?** **No.** El AUC solo depende del **orden** de los puntajes (es invariante a cualquier transformación monótona); log loss y Brier evalúan además la **calibración**, es decir, si p = 0,2 corresponde a ~20 % de contrataciones. Ejemplo claro: Naive Bayes tiene AUC {{nb_auc}} (similar a la logística con 10 columnas, {{log10_auc}}) pero el peor log loss y Brier. Del mismo modo, el árbol de profundidad 6 supera al de profundidad 3 en AUC pero es peor en log loss. Las curvas de calibración muestran que boosting, RF y la logística siguen la diagonal razonablemente, mientras que Naive Bayes se aleja. Para decidir con umbrales y costos (pregunta 3), importan las probabilidades, no solo el orden.'''),
    ('md', r'''## Pregunta 3. Evaluación y decisión (10 puntos)'''),
    ('md', r'''### 3(a) (4 puntos)

Con las probabilidades OOF del modelo elegido, evalúen los umbrales 0.05; 0.10; ...; 0.95. Predigan contratación si p es mayor o igual al umbral. Elijan el que minimiza el costo hipotético C(t) = FP(t) + 5 FN(t), donde FP es una alerta de contratación que no ocurre y FN una contratación no detectada. Presenten una tabla o curva e interpreten ambos errores.'''),
    ('code', r'''p_oof = oof[MODELO_ELEGIDO]
umbrales = np.round(np.arange(0.05, 0.951, 0.05), 2)
filas = []
for t in umbrales:
    pred = (p_oof >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    filas.append({'umbral': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                  'sensibilidad': tp / (tp + fn), 'especificidad': tn / (tn + fp),
                  'contactados': tp + fp, 'costo': fp + 5 * fn})
tabla_costos = pd.DataFrame(filas)
tabla_costos['costo_por_registro'] = tabla_costos.costo / len(y)
print(tabla_costos.round(3).to_string(index=False))
UMBRAL = float(tabla_costos.loc[tabla_costos.costo.idxmin(), 'umbral'])
print(f'\nUmbral elegido (mínimo costo OOF): {UMBRAL}')
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
_pub('sens_umbral_050', _num(100 * _fila(0.50).sensibilidad, 0))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tabla_costos.umbral, tabla_costos.costo, marker='o', label='C(t) = FP + 5·FN')
ax.plot(tabla_costos.umbral, tabla_costos.FP, ls='--', label='FP')
ax.plot(tabla_costos.umbral, 5 * tabla_costos.FN, ls=':', label='5·FN')
ax.axvline(UMBRAL, color='gray', lw=1); ax.set_xlabel('Umbral t'); ax.set_ylabel('Costo OOF')
ax.set_title(f'Costo hipotético OOF – {MODELO_ELEGIDO}'); ax.legend(); plt.tight_layout(); plt.show()'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

Con las probabilidades OOF del {{modelo_elegido_nombre}}, el costo C(t) = FP(t) + 5·FN(t) se minimiza en **t = {{umbral_elegido}}** (C = {{costo_umbral}}; C/n = {{costo_n_umbral}}), con {{tp_umbral}} TP, {{fp_umbral}} FP, {{fn_umbral}} FN y {{tn_umbral}} TN: sensibilidad {{sens_umbral}} % y especificidad {{espec_umbral}} %, contactando a {{contactados_umbral}} de {{n_registros_entrenamiento}} registros ({{pct_contactados_umbral}} %). Los umbrales {{umbral_020}} ({{costo_020}}) y {{umbral_010}} ({{costo_010}}) le siguen.

Como referencia: no contactar a nadie cuesta 5·{{n_positivos}} = {{costo_no_contactar}}, y contactar a todos {{costo_contactar_todos}}. El umbral {{umbral_050}} cuesta {{costo_050}} (solo detecta el {{sens_umbral_050}} % de los contratantes).

**Interpretación de los errores:**
- **FP (falsa alerta):** llamar a un cliente que no contratará; cuesta una llamada poco productiva (costo 1). Bajar el umbral los aumenta rápidamente ({{fp_010}} FP con t = {{umbral_010}}; {{fp_005}} con t = {{umbral_005}}).
- **FN (oportunidad perdida):** no llamar a un cliente que sí contrataría; cuesta 5 veces más. Subir el umbral los aumenta.

Como un FN vale 5 FP, conviene contactar a un cliente si su probabilidad supera aprox. 1/(1+5) ≈ 0,167 (umbral óptimo de Bayes para probabilidades bien calibradas). El óptimo empírico, {{umbral_elegido}}, es consistente con ese valor y con que el modelo esté razonablemente calibrado. Por eso el umbral es mucho menor que 0,50: el costo asimétrico justifica tolerar más falsas alertas. La curva de costo es bastante plana entre {{umbral_elegido}} y {{umbral_025}}, lo que da cierta robustez a la elección.'''),
    ('md', r'''### 3(b) (6 puntos)

Reajusten el modelo elegido con todo el entrenamiento y evalúen en `clientes_prueba.csv`. Informen log loss, Brier, AUC y curva ROC. Comparen el umbral elegido con 0.50 mediante matriz de confusión, sensibilidad, especificidad y C/n; incluyan el modelo de referencia. ¿Qué recomendarían al banco? ¿Qué permiten concluir los resultados y qué falta verificar para una campaña futura?'''),
    ('code', r'''# Registro de decisiones tomadas SOLO con OOF (antes de mirar la prueba)
print('Modelo elegido:', MODELO_ELEGIDO, '| umbral elegido:', UMBRAL)
prueba = pd.read_csv(CARPETA / 'clientes_prueba.csv')
datos_prueba = construir_variables(prueba)
y_test = datos_prueba[RESPUESTA].to_numpy()
print('Prueba:', prueba.shape, '| tasa de contratación en prueba: %.4f' % y_test.mean())

constructores = {'Logística (todas)': logit, **{k: v[0] for k, v in modelos.items()}}
prep_final = nuevo_preparador().fit(datos)
X_full, X_test = prep_final.transform(datos), prep_final.transform(datos_prueba)
if MODELO_ELEGIDO == 'Logística (10 F-ANOVA)':
    sel_final = SelectKBest(f_classif, k=10).fit(X_full, y)
    X_full, X_test = sel_final.transform(X_full), sel_final.transform(X_test)
    constructor = logit
else:
    constructor = constructores[MODELO_ELEGIDO]
modelo_final = constructor(X_full.shape[1]).fit(X_full, y)
p_test = modelo_final.predict_proba(X_test)[:, 1]
p_ref = np.full(len(y_test), y.mean())   # referencia: proporción de todo el entrenamiento

met = pd.DataFrame({MODELO_ELEGIDO: metricas(y_test, p_test), 'Referencia': metricas(y_test, p_ref)}).T
met.loc['Referencia', 'auc'] = 0.5
print(met.round(4))

def resumen_umbral(p, t, nombre):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {'regla': nombre, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'sensibilidad': tp / (tp + fn), 'especificidad': tn / (tn + fp),
            'contactados': tp + fp, 'C': fp + 5 * fn, 'C/n': (fp + 5 * fn) / len(y_test)}
comp = pd.DataFrame([resumen_umbral(p_test, UMBRAL, f'{MODELO_ELEGIDO}, t={UMBRAL}'),
                     resumen_umbral(p_test, 0.5, f'{MODELO_ELEGIDO}, t=0.50'),
                     resumen_umbral(p_ref, UMBRAL, f'Referencia, t={UMBRAL} (p={y.mean():.3f}: no contacta a nadie)'),
                     resumen_umbral(p_ref, 0.5, 'Referencia, t=0.50 (no contacta a nadie)'),
                     resumen_umbral(np.ones_like(p_ref), 0.5, 'Contactar a todos (comparación)')])
print(comp.round(4).to_string(index=False))
for _, r in comp.iloc[:2].iterrows():
    print(f"\nMatriz de confusión – {r.regla}\n", pd.DataFrame([[r.TN, r.FP], [r.FN, r.TP]],
          index=['real 0', 'real 1'], columns=['pred 0', 'pred 1']))

fpr, tpr, _ = roc_curve(y_test, p_test)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f'{MODELO_ELEGIDO} (AUC = {roc_auc_score(y_test, p_test):.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Referencia (AUC = 0.5)')
for t, mk in [(UMBRAL, 'o'), (0.5, 's')]:
    r = resumen_umbral(p_test, t, '')
    ax.plot(1 - r['especificidad'], r['sensibilidad'], mk, ms=9, label=f't = {t}')
ax.set_xlabel('1 - especificidad'); ax.set_ylabel('Sensibilidad'); ax.set_title('Curva ROC en prueba'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# Lift: tasa de contratación por decil de probabilidad en prueba
dec = pd.qcut(pd.Series(p_test).rank(method='first', ascending=False), 10, labels=range(1, 11))
por_decil = pd.DataFrame({'y': y_test, 'decil': dec}).groupby('decil', observed=True)['y'].agg(['size', 'mean']).round(3)
print('\nTasa real de contratación por decil de probabilidad (1 = más alta):')
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
_pub('test_se_sens', _num(100 * np.sqrt(_g15.sensibilidad * (1 - _g15.sensibilidad) / int(y_test.sum())), 1))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**Decisiones registradas antes de abrir la prueba:** modelo {{modelo_elegido_nombre}} y umbral {{umbral_elegido}}. El modelo se reajustó con los {{n_registros_entrenamiento}} registros (preparación aprendida en todo el entrenamiento) y se evaluó una sola vez en los {{n_test}} de prueba.

| | Log loss | Brier | AUC |
|---|---|---|---|
| Gradient Boosting | **{{gb_test_log_loss}}** | **{{gb_test_brier}}** | **{{gb_test_auc}}** |
| Referencia (p = {{p_ref}}) | {{ref_test_log_loss}} | {{ref_test_brier}} | {{ref_test_auc}} |

Las métricas de prueba son muy parecidas a las OOF ({{gb_log_loss}} / {{gb_brier}} / {{gb_auc}}), lo que indica que la validación cruzada estimó bien el desempeño y no hubo sobreajuste en la selección.

| Regla | TP | FP | FN | TN | Sens. | Espec. | Contactados | C | C/n |
|---|---|---|---|---|---|---|---|---|---|
| GB, t = {{umbral_elegido}} | {{test_tp_015}} | {{test_fp_015}} | {{test_fn_015}} | {{test_tn_015}} | {{test_sens_015}} % | {{test_espec_015}} % | {{test_contactados_015}} | {{test_costo_015}} | **{{test_cn_015}}** |
| GB, t = {{umbral_050}} | {{test_tp_050}} | {{test_fp_050}} | {{test_fn_050}} | {{test_tn_050}} | {{test_sens_050}} % | {{test_espec_050}} % | {{test_contactados_050}} | {{test_costo_050}} | {{test_cn_050}} |
| Referencia (t = {{umbral_elegido}} o {{umbral_050}}) | 0 | 0 | {{test_n_positivos}} | {{test_n_test_menos}} | 0 % | 100 % | 0 | {{test_costo_ref}} | {{test_cn_ref}} |
| Contactar a todos | {{test_n_positivos}} | {{test_n_negativos}} | 0 | 0 | 100 % | 0 % | {{n_test}} | {{test_costo_todos}} | {{test_cn_todos}} |

La referencia predice p = {{p_ref}} para todos: con ambos umbrales no contacta a nadie (costo {{test_costo_ref}}). El umbral {{umbral_elegido}} reduce el costo en {{test_reduccion_pct_050}} % respecto de {{umbral_050}} y en {{test_reduccion_pct_ref}} % respecto de la referencia: detecta {{test_tp_015}} de los {{test_n_positivos}} contratantes llamando solo al {{test_pct_cartera}} % de la cartera, con una tasa de éxito entre los contactados de {{test_tasa_exito}} % ({{test_tp_015}}/{{test_contactados_015}}), casi {{test_veces_tasa_base}} veces la tasa base ({{tasa_real}} %). En el decil de mayor probabilidad la tasa real de contratación es {{test_decil1_tasa}} %.

**Recomendación al banco:** priorizar la campaña ordenando a los clientes por la probabilidad estimada y contactar a quienes tengan p ≥ {{umbral_elegido}} (o, si la capacidad es limitada, comenzar por los deciles superiores). Los perfiles con mayor probabilidad se relacionan con éxito en campañas anteriores, contacto previo, canal celular y ciertos meses.

**Qué permiten concluir:** con la información disponible *antes* de la llamada el modelo discrimina moderadamente (AUC {{gb_test_auc}}) y, bajo el costo hipotético 1:5, una regla con t = {{umbral_elegido}} es claramente mejor que no usar modelo, que contactar a todos o que usar 0,50.

**Qué falta verificar:**
- Los **costos reales** de una llamada y el valor de un depósito (la razón 1:5 es hipotética; el umbral óptimo cambia con ella) y la **capacidad** del call center.
- **Validez temporal:** los datos son de campañas pasadas (2008–2010, con contexto macroeconómico particular). `mes` y `dia` reflejan cuándo se llamó, no necesariamente algo controlable a futuro; y `contactos_campana` incluye la llamada registrada, así que para predecir antes del primer contacto de una campaña nueva habría que redefinirla.
- **Causalidad:** el modelo predice quién contrata *si se le llama*, no el efecto incremental de llamar (haría falta un experimento con grupo control o modelos de *uplift*).
- Incertidumbre: con {{test_n_positivos}} positivos en prueba, la sensibilidad tiene un error estándar de ~{{test_se_sens}} puntos; conviene un piloto, monitoreo de calibración y recalibración periódica.'''),
    ('md', r'''## Pregunta 4. Componentes principales y agrupamiento (8 puntos)'''),
    ('md', r'''### 4(a) (3 puntos)

Realicen PCA. Informen la varianza explicada por PC1 y PC2, su suma y las cargas; grafiquen las observaciones en ese plano. ¿Qué información resume la proyección y cuál omite?'''),
    ('code', r'''VARS4 = ['edad', 'saldo', 'contactos_campana', 'contactos_previos']
Z = (entrenamiento[VARS4] - entrenamiento[VARS4].mean()) / entrenamiento[VARS4].std(ddof=1)   # desviación muestral
pca = PCA(random_state=SEMILLA).fit(Z)
scores = pca.transform(Z)
var = pd.DataFrame({'varianza': pca.explained_variance_, 'prop_explicada': pca.explained_variance_ratio_,
                    'acumulada': np.cumsum(pca.explained_variance_ratio_)}, index=[f'PC{i+1}' for i in range(4)])
print(var.round(4))
print('PC1 + PC2 = %.4f' % pca.explained_variance_ratio_[:2].sum())
cargas = pd.DataFrame(pca.components_.T, index=VARS4, columns=var.index)
print('\nCargas:'); print(cargas.round(4))
print('\nCorrelaciones entre las 4 variables:'); print(Z.corr().round(3))

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
_pub('pca_max_corr_par', f'{VARS4[_i]} y {VARS4[_j]}')

fig, ax = plt.subplots(figsize=(7, 5.5))
for clase, col in [(0, 'tab:blue'), (1, 'tab:orange')]:
    m = entrenamiento[RESPUESTA].to_numpy() == clase
    ax.scatter(scores[m, 0], scores[m, 1], s=8, alpha=.45, c=col, label=f'contrata = {clase}')
for v in VARS4:   # biplot: flechas de cargas
    ax.arrow(0, 0, 3 * cargas.loc[v, 'PC1'], 3 * cargas.loc[v, 'PC2'], color='k', head_width=.12)
    ax.text(3.3 * cargas.loc[v, 'PC1'], 3.3 * cargas.loc[v, 'PC2'], v, fontsize=9)
ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
ax.set_title('Observaciones en el plano PC1–PC2'); ax.legend(); plt.tight_layout(); plt.show()'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

Se aplicó PCA sobre `edad`, `saldo`, `contactos_campana` y `contactos_previos` estandarizadas (desviación muestral).

| | Varianza | % explicado | Acumulado |
|---|---|---|---|
| PC1 | {{pca_pc1_var}} | {{pca_pc1_pct}} % | {{pca_pc1_acum}} % |
| PC2 | {{pca_pc2_var}} | {{pca_pc2_pct}} % | **{{pca_pc2_acum}} %** |
| PC3 | {{pca_pc3_var}} | {{pca_pc3_pct}} % | {{pca_pc3_acum}} % |
| PC4 | {{pca_pc4_var}} | {{pca_pc4_pct}} % | 100 % |

**Cargas:** PC1 = {{pca_carga_edad_pc1}}·edad {{pca_carga_saldo_pc1}}·saldo {{pca_carga_contactos_campana_pc1}}·contactos_campana {{pca_carga_contactos_previos_pc1}}·contactos_previos → contrasta clientes de **mayor edad y saldo** (y con historial previo) contra los muy contactados en esta campaña. PC2 = {{pca_carga_edad_pc2}}·edad {{pca_carga_saldo_pc2}}·saldo {{pca_carga_contactos_campana_pc2}}·contactos_campana {{pca_carga_contactos_previos_pc2}}·contactos_previos → contrasta **historial de campañas previas** contra intensidad de contactos en la campaña actual (y edad/saldo). (El signo de cada componente es arbitrario.)

**Qué resume y qué omite:** las cuatro variables están casi **incorrelacionadas** (la mayor correlación es {{pca_max_corr}} entre {{pca_max_corr_par}}), por lo que no hay una estructura de baja dimensión que comprimir: cada componente explica cerca del 25 % esperado bajo independencia, y el plano PC1–PC2 conserva solo el **{{pca_sum_pct}} %** de la varianza total; el {{pca_omitido_pct}} % restante (PC3 y PC4) se pierde, así que las distancias en el gráfico pueden engañar. Además, PCA es lineal y sensible a valores extremos: el gráfico está dominado por una nube densa con algunos puntos muy alejados (saldos y contactos extremos). La proyección **no usa la respuesta**: las clases contrata = 0/1 se superponen casi por completo, por lo que tampoco resume la información relevante para predecir contratación, ni incluye las variables categóricas (mes, resultado previo, canal), que son las más predictivas.'''),
    ('md', r'''### 4(b) (5 puntos)

Ajusten K-means sobre las cuatro variables estandarizadas con k = 2, 3 y 4 y al menos 30 inicializaciones. Comparen WSS y silueta media y elijan el k de mayor silueta. Informen tamaños, describan perfiles y representen los grupos en el plano PCA. Calculen la proporción de contratación dentro de cada grupo y expliquen por qué los grupos no equivalen a clases predichas.'''),
    ('code', r'''resumen_k = []; ajustes = {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, n_init=30, random_state=SEMILLA).fit(Z)
    ajustes[k] = km
    resumen_k.append({'k': k, 'WSS': km.inertia_, 'silueta_media': silhouette_score(Z, km.labels_),
                      'tamaños': sorted(np.bincount(km.labels_).tolist(), reverse=True)})
resumen_k = pd.DataFrame(resumen_k); print(resumen_k.round(4).to_string(index=False))
K = int(resumen_k.loc[resumen_k.silueta_media.idxmax(), 'k'])
print('\nk elegido (mayor silueta):', K)
grupo = ajustes[K].labels_
perfil = entrenamiento[VARS4 + [RESPUESTA]].assign(grupo=grupo).groupby('grupo').agg(
    n=(RESPUESTA, 'size'), edad=('edad', 'mean'), saldo=('saldo', 'mean'), saldo_mediana=('saldo', 'median'),
    contactos_campana=('contactos_campana', 'mean'), contactos_previos=('contactos_previos', 'mean'),
    pct_con_contacto_previo=('contactos_previos', lambda s: (s > 0).mean()), tasa_contratacion=(RESPUESTA, 'mean'))
print('\nPerfiles (valores originales):'); print(perfil.round(3).to_string())
print('\nCentroides (escala estandarizada):')
print(pd.DataFrame(ajustes[K].cluster_centers_, columns=VARS4).round(2))
print('Tasa global de contratación: %.4f' % entrenamiento[RESPUESTA].mean())

fig, ax = plt.subplots(figsize=(7, 5.5))
for g in range(K):
    m = grupo == g
    ax.scatter(scores[m, 0], scores[m, 1], s=8, alpha=.5, label=f'Grupo {g} (n={m.sum()})')
cen = pca.transform(pd.DataFrame(ajustes[K].cluster_centers_, columns=VARS4))
ax.scatter(cen[:, 0], cen[:, 1], c='k', marker='X', s=150, label='Centroides')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(f'K-means (k = {K}) en el plano PCA'); ax.legend()
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
_pub('k_max_tasa', _num(100 * perfil.tasa_contratacion.max(), 1))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

| k | WSS | Silueta media | Tamaños |
|---|---|---|---|
| 2 | {{k2_wss}} | {{k2_sil}} | {{k2_tamanos}} |
| 3 | {{k3_wss}} | {{k3_sil}} | {{k3_tamanos}} |
| 4 | {{k4_wss}} | **{{k4_sil}}** | {{k4_tamanos}} |

El WSS siempre baja al aumentar k (por construcción), por lo que no sirve por sí solo para elegir; la silueta media es mayor en **k = {{k_elegido}}**, que se elige (K-means con 30 inicializaciones, semilla 2026). Una silueta de {{silueta_k}} indica una estructura **débil a moderada**, coherente con variables casi incorrelacionadas.

**Perfiles (medias en escala original):**
- **Grupo 1 – Jóvenes, saldo bajo, nuevos** (n = {{g1_n}}): edad {{g1_edad}}, saldo medio {{g1_saldo}} € (mediana {{g1_saldo_med}}), {{g1_contactos}} contactos, y solo el {{g1_pct_previo}} % tiene campañas previas. Contratación **{{g1_tasa}} %**.
- **Grupo 2 – Mayores con más saldo** (n = {{g2_n}}): edad {{g2_edad}}, saldo medio {{g2_saldo}} € (mediana {{g2_saldo_med}}), sin historial relevante ({{g2_pct_previo}} % con contacto previo). Contratación **{{g2_tasa}} %**.
- **Grupo 0 – Con historial de campañas previas** (n = {{g0_n}}): {{g0_pct_previo}} % con contacto previo, {{g0_contactos_previos}} contactos previos en promedio. Contratación **{{g0_tasa}} %** (más del doble de la tasa global, {{tasa_global}} %).
- **Grupo 3 – Muy insistidos en esta campaña** (n = {{g3_n}}): {{g3_contactos}} contactos en la campaña en promedio. Contratación **{{g3_tasa}} %**.

En el plano PCA, los grupos 1 y 2 se separan principalmente por PC1 (edad/saldo), y los grupos pequeños 0 y 3 aparecen hacia los extremos de PC2 (historial previo vs. contactos en la campaña), con superposición en la nube central (recordar que el plano solo muestra el {{pca_sum_pct}} % de la varianza).

**¿Por qué los grupos no son clases predichas?** K-means es **no supervisado**: forma grupos minimizando distancias en las 4 variables, **sin usar `contrata`**. Los grupos describen perfiles de clientes, no fronteras de decisión entre contratantes y no contratantes: dentro de cada grupo conviven ambas clases (el grupo más favorable llega al {{k_max_tasa}} % de contratación, por lo que una regla "grupo → clase" predeciría siempre "no contrata"). Las proporciones de contratación por grupo se calcularon **después**, como descripción; son útiles para segmentar (p. ej. priorizar el grupo con historial previo y evitar insistir con quien ya recibió muchas llamadas), pero no reemplazan un modelo supervisado evaluado con métricas fuera de muestra.'''),
    ('md', r'''## Pregunta 5. Detección de anomalías (6 puntos)'''),
    ('md', r'''### 5(a) (3 puntos)

Calculen la distancia de Mahalanobis al cuadrado y usen como corte el cuantil 0.99 de una chi-cuadrado con 4 grados de libertad. Informen el corte, la cantidad y el porcentaje de registros marcados. Identifiquen los dos de mayor distancia y presenten sus valores originales.'''),
    ('code', r'''S = np.cov(Z.to_numpy(), rowvar=False, ddof=1)       # = matriz de correlación muestral
S_inv = np.linalg.inv(S)
Zc = Z.to_numpy() - Z.to_numpy().mean(axis=0)          # media 0 por construcción
d2 = np.einsum('ij,jk,ik->i', Zc, S_inv, Zc)
corte = stats.chi2.ppf(0.99, df=4)
marcados = d2 > corte
print(f'Corte chi2(0.99; 4) = {corte:.4f}')
print(f'Registros marcados: {marcados.sum()} de {len(d2)} ({100 * marcados.mean():.2f}%)')
anom = entrenamiento.assign(d2=d2)
top2 = anom.sort_values('d2', ascending=False).head(2)
print('\nDos registros con mayor distancia (valores originales):')
print(top2[['id_cliente'] + VARS4 + ['d2', RESPUESTA]].to_string(index=False))
print('\nValores estandarizados de esos registros:')
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
    _pub(f'anom{_i + 1}_contrata', int(_row.contrata))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

Con las cuatro variables estandarizadas, la distancia de Mahalanobis al cuadrado es d² = zᵀ S⁻¹ z, con S la matriz de covarianza muestral de las variables estandarizadas (= matriz de correlación).

- **Corte:** χ²₀,₉₉(4) = **{{corte_chi2}}**.
- **Registros marcados:** **{{n_marcados}} de {{n_registros_entrenamiento}} ({{pct_marcados}} %)**.
- **Dos de mayor distancia (valores originales):**

| id_cliente | edad | saldo | contactos_campana | contactos_previos | d² | contrata |
|---|---|---|---|---|---|---|
| {{anom1_id}} | {{anom1_edad}} | {{anom1_saldo}} | {{anom1_contactos_campana}} | {{anom1_contactos_previos}} | {{anom1_d2}} | {{anom1_contrata}} |
| {{anom2_id}} | {{anom2_edad}} | {{anom2_saldo}} | {{anom2_contactos_campana}} | {{anom2_contactos_previos}} | {{anom2_d2}} | {{anom2_contrata}} |'''),
    ('md', r'''### 5(b) (3 puntos)

¿Por qué son inusuales los dos registros identificados? ¿Esto demuestra que contienen errores? ¿El corte utilizado garantiza marcar al 1% de la muestra? Fundamenten e indiquen qué revisarían antes de eliminar un registro.'''),
    ('code', r'''# Contexto para interpretar: percentiles y posición de los dos registros
desc = entrenamiento[VARS4].describe(percentiles=[.5, .9, .99]).T[['mean', '50%', '90%', '99%', 'max']]
print(desc.round(1))
for idx in top2.index:
    r = entrenamiento.loc[idx]
    print(f"\nid_cliente {r.id_cliente}: " + ', '.join(
        f"{v} = {r[v]} (percentil {stats.percentileofscore(entrenamiento[v], r[v], kind='weak'):.2f})" for v in VARS4))
    print('   Registro completo:', entrenamiento.loc[idx, ['ocupacion', 'edad', 'saldo', 'mes', 'contactos_campana',
          'dias_previos', 'contactos_previos', 'resultado_previo', 'contrata']].to_dict())
# ¿Las variables siguen una normal multivariada? (supuesto del corte chi-cuadrado)
print('\nAsimetría de las variables:', entrenamiento[VARS4].skew().round(2).to_dict())
print('Curtosis (exceso):', entrenamiento[VARS4].kurt().round(2).to_dict())
print('Proporción esperada sobre el corte bajo normalidad: 1.00%% | observada: %.2f%%' % (100 * marcados.mean()))
print('Contribución de cada variable al exceso: registros marcados con |z| > 3 por variable:')
print((Z[marcados].abs() > 3).sum().to_string())
print('\nTasa de contratación: marcados %.3f vs no marcados %.3f' % (entrenamiento.contrata[marcados].mean(), entrenamiento.contrata[~marcados].mean()))

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
_pub('tasa_no_marcados', _num(100 * entrenamiento.contrata[~marcados].mean(), 1))'''),
    ('md', r'''**Respuesta e interpretación del grupo:**

**¿Por qué son inusuales?**
- **id {{anom1_id}}:** su **saldo de {{anom1_saldo}} €** es el máximo de la muestra, a **{{z1_saldo}} desviaciones estándar** de la media (la mediana es {{saldo_mediana}} € y el percentil 99 es ≈ {{saldo_p99}} €). Las otras variables son normales ({{anom1_edad}} años, jubilado, {{anom1_contactos_campana}} contacto). Como las variables están casi incorrelacionadas, la distancia se debe casi por completo a ese saldo ({{z1_saldo}}² ≈ {{d2_1}}).
- **id {{anom2_id}}:** recibió **{{anom2_contactos_campana}} contactos** en la campaña, el máximo, a **{{z2_contactos_campana}} desviaciones** de la media (la mediana es {{cc_mediana}} y el percentil 99 es {{cc_p99}}). Edad, saldo y contactos previos son típicos.

En ambos casos es una sola variable extrema, no una combinación inusual de variables.

**¿Demuestra que son errores?** **No.** Mahalanobis solo indica que un registro está lejos del centro de los datos según su estructura de covarianza. Un jubilado con un saldo alto (p. ej. una indemnización o la venta de una propiedad) es plausible, y un cliente llamado {{anom2_contactos_campana}} veces puede reflejar una política de insistencia o rellamadas automáticas. Son atípicos **estadísticos**, que pueden ser genuinos (y quizás valiosos) o errores de registro; la distancia no distingue entre ambos.

**¿El corte garantiza marcar al 1 %?** **No.** El cuantil 0,99 de χ²(4) solo implica 1 % esperado **si los datos fueran normales multivariados** (y con media y covarianza conocidas). Aquí las variables son muy asimétricas y de colas pesadas (asimetría: saldo {{skew_saldo}}, contactos_campana {{skew_contactos_campana}}, contactos_previos {{skew_contactos_previos}}; curtosis en exceso {{kurt_saldo}}, {{kurt_contactos_campana}} y {{kurt_contactos_previos}}), son discretas (conteos con muchos ceros) y la media y covarianza se estiman con los mismos atípicos que se buscan. Por eso se marca el **{{pct_marcados}} %**, unas {{factor_exceso}} veces el 1 % esperado. El corte es un umbral de referencia, no una tasa garantizada; con datos así sería preferible un estimador robusto (MCD) o transformar las variables (p. ej. log del saldo) antes.

**Antes de eliminar un registro revisaríamos:**
1. Su **consistencia con otras fuentes** (sistema contable del banco para el saldo; registro del call center para los contactos) y con las demás columnas del mismo cliente.
2. Si el valor es **posible** según el dominio y las reglas de negocio (no hay valores imposibles aquí).
3. **Su efecto en el análisis:** comparar resultados con y sin el registro; los árboles y ensambles son robustos a valores extremos en predictores, y la transformación log o el *winsorizing* pueden limitar su influencia sin descartar información.
4. Si los atípicos se relacionan con la respuesta (los marcados contratan {{tasa_marcados}} % vs {{tasa_no_marcados}} %): eliminarlos podría sesgar el modelo.
5. Documentar la decisión. Solo eliminaríamos registros con errores comprobados; de lo contrario, se conservan (además, el enunciado pide conservar todas las filas).'''),
    ('md', r'''## Criterios de evaluación

Se evaluarán procedimiento, resultados e interpretación. Se asignará crédito parcial sin descontar dos veces un mismo error arrastrado. Se aceptan diferencias numéricas y ganadores distintos cuando la implementación y la regla de selección estén justificadas.'''),
    ('code', r'''# --- Celda técnica de recolección ------------------------------------------
# Publica todos los valores capturados con _pub() para poder inyectarlos en el
# texto del informe. Esta celda NO forma parte del notebook entregado.
for _clave in sorted(R):
    print(f'@@{_clave}={R[_clave]}@@')
'''),]

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


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
    print("Marcadores presentes:", len(re.findall(r"\{\{\w+\}\}", "\n".join(c.source for c in nb.cells if c.cell_type == "markdown"))))
