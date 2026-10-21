from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# Title/header
md("""# Taller #3 — Respuesta a una campaña bancaria

Diplomado en Data Science UC 2026 — Aprendizaje Automático

**Grupo:** 36  
**Integrantes (3 a 5):** Nadia Lell, Jaysson Leon, Kelly Martinez, Matias Reyes 
**Lenguaje:** Python  
**Puntaje máximo: 60. Nota = 1 + P/10.**
""")

md("""## El caso

Un banco portugués realiza campañas telefónicas para ofrecer depósitos a plazo. Contactar a toda la cartera tiene un costo y muchos clientes no contratan el producto. El equipo comercial quiere anticipar qué registros tienen mayor probabilidad de contratación y comprender los perfiles de la campaña. Una predicción favorable que no se concreta implica un contacto poco productivo; una predicción desfavorable puede dejar fuera una oportunidad.

Como equipo de Data Science, deberán preparar los datos, comparar modelos y recomendar a qué clientes contactar. La predicción se realiza antes de la llamada registrada. El análisis de grupos y de registros inusuales complementará su recomendación.

Se utiliza la muestra pública `bank.csv` de Bank Marketing, con 4521 registros. Cada fila contiene información de un cliente y sus contactos, junto con el resultado de contratación.
""")

md("""## Indicaciones de trabajo

Trabajen con `clientes_entrenamiento.csv` y reserven `clientes_prueba.csv` para 3(b). Conserven las filas, fijen semilla 2026 y consideren `contrata` = 1 como clase positiva.

Usen los cinco pliegues entregados para la validación cruzada. Aprendan las transformaciones y la selección de variables solo con las filas de ajuste de cada pliegue. Reúnan predicciones fuera de pliegue (OOF): probabilidades para los modelos probabilísticos y puntajes de decisión para SVM.

El modelo de referencia predice la proporción positiva de las filas de ajuste. Para la evaluación final, calculen esa proporción con todo el conjunto de entrenamiento.
""")

md("""## Datos

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
""")

md("""## Preparación

Funciones de apoyo: OneHotEncoder, StandardScaler, f_classif/SelectKBest, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, SVC y decision_function, GaussianNB, PCA y KMeans.
""")

code("""%matplotlib inline
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(2026)
CARPETA = Path('.')
entrenamiento = pd.read_csv(CARPETA / 'clientes_entrenamiento.csv')
print('Python:', sys.version.split()[0])
print('pandas:', pd.__version__, '| numpy:', np.__version__, '| sklearn:', sklearn.__version__)
print('scipy:', __import__('scipy').__version__, '| matplotlib:', __import__('matplotlib').__version__)
print('Forma entrenamiento:', entrenamiento.shape)
""")

# Pregunta 1
md("## Pregunta 1. Problema, preparaci\u00f3n y representaci\u00f3n (12 puntos)")

md("""### 1(a) (3 puntos)

Identifiquen la tarea de aprendizaje, la unidad de observaci\u00f3n y la respuesta. \u00bfPor qu\u00e9 `duracion` no puede usarse para predecir antes de la llamada? Excl\u00fayanla de los predictores, junto con `id_cliente`, `pliegue` y `contrata`.
""")

code("""excluir = ['duracion', 'id_cliente', 'pliegue', 'contrata']
predictores = [c for c in entrenamiento.columns if c not in excluir]
print('Predictores retenidos:', predictores)
print('Columnas excluidas:', excluir)
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

La tarea es de **clasificaci\u00f3n binaria supervisada**: estimar la probabilidad de que un cliente contrate el dep\u00f3sito a plazo (`contrata` = 1) a partir de informaci\u00f3n disponible **antes** de la llamada registrada. La unidad de observaci\u00f3n es cada **cliente/contacto** de la campaña (una fila del archivo). La respuesta es `contrata` (1 = contrata, 0 = no contrata).

`duracion` no puede usarse porque es la duraci\u00f3n de la **llamada ya realizada**; en el momento de priorizar a qui\u00e9n contactar esa informaci\u00f3n a\u00fan no existe (ser\u00eda fuga de informaci\u00f3n, *data leakage*). Adem\u00e1s, su relaci\u00f3n con la contrataci\u00f3n es casi determinista y sobreestimar\u00eda artificialmente el desempe\u00f1o. Por eso se excluye junto con `id_cliente` (identificador sin valor predictivo), `pliegue` (solo para validaci\u00f3n) y `contrata` (respuesta).
""")

md("""### 1(b) (4 puntos)

Revisen tipos de variables, faltantes, categor\u00edas unknown, duplicados y balance de clases. Interpreten los valores negativos de `saldo` y el c\u00f3digo -1 de `dias_previos`. \u00bfSer\u00eda suficiente la exactitud de predecir siempre la clase mayoritaria? Fundamenten.
""")

code("""import pandas as pd
df = entrenamiento.copy()
print('Tipos de variables:')
print(df.dtypes)
print('\\nValores faltantes por columna:')
print(df.isna().sum())
print('\\nDuplicados de fila:', df.duplicated().sum())
print('Duplicados de id_cliente:', df['id_cliente'].duplicated().sum())
print('\\nBalance de clases (contrata):')
print(df['contrata'].value_counts())
print(df['contrata'].value_counts(normalize=True).round(4))
print('\\nCategor\u00edas unknown por variable:')
for c in ['ocupacion','estado_civil','educacion','mora','hipoteca','prestamo','canal','mes','resultado_previo']:
    print(f'  {c}: {(df[c]=="unknown").sum()}')
print('\\nRango saldo:', df['saldo'].min(), df['saldo'].max())
print('saldo negativo (n):', (df['saldo']<0).sum())
print('\\ndias_previos = -1 (n):', (df['dias_previos']==-1).sum())
print('contactos_previos = 0 (n):', (df['contactos_previos']==0).sum())
print('\\nExactitud de predecir siempre la clase mayoritaria:', round(df['contrata'].value_counts(normalize=True).max(),4))
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

No hay valores faltantes ni filas duplicadas. Las variables son num\u00e9ricas (`edad`, `saldo`, `dia`, `duracion`, `contactos_campana`, `dias_previos`, `contactos_previos`) y categ\u00f3ricas (`ocupacion`, `estado_civil`, `educacion`, `mora`, `hipoteca`, `prestamo`, `canal`, `mes`, `resultado_previo`). Existen categor\u00edas `unknown` (p. ej. `canal` 1088, `educacion` 154, `ocupacion` 35, `resultado_previo` 2979), que se mantienen como categor\u00eda propia y no se imputan. El balance es desbalanceado: 88,47 % no contrata y 11,53 % contrata.

Los saldos negativos son plausibles (sobregiros o deudas) y no deben eliminarse autom\u00e1ticamente; contienen informaci\u00f3n sobre la situaci\u00f3n financiera. El c\u00f3digo -1 en `dias_previos` significa que no hubo contacto previo; equivale a `contactos_previos = 0` y `resultado_previo = unknown` (2979 casos). Se conserva mediante un indicador `contactado_antes` y se reemplaza -1 por 0.

Predecir siempre la clase mayoritaria dar\u00eda una exactitud de 0,8847, pero **no es suficiente**: tendr\u00eda sensibilidad 0 para la clase positiva, es decir, nunca detectar\u00eda a un cliente que contrata. Dado que el costo de un falso negativo (oportunidad perdida) se valora 5 veces m\u00e1s que un falso positivo, un modelo que ignore la minor\u00eda es in\u00fatil para la campa\u00f1a.
""")

md("""### 1(c) (5 puntos)

Construyan dos variables: `contactado_antes` = 1 si `dias_previos` es distinto de -1, y `log_contactos_previos` = log(1 + `contactos_previos`). Expliquen su sentido. Reemplacen -1 por 0 en `dias_previos` conservando el indicador, mantengan unknown como categor\u00eda y codifiquen las variables categ\u00f3ricas mediante one-hot. Describan y apliquen el escalamiento necesario para cada m\u00e9todo.
""")

code("""from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def base_features(df):
    df = df.copy()
    df['contactado_antes'] = (df['dias_previos'] != -1).astype(int)
    df['log_contactos_previos'] = np.log1p(df['contactos_previos'])
    df['dias_previos'] = df['dias_previos'].replace(-1, 0)
    return df

entrenamiento = base_features(entrenamiento)
print(entrenamiento[['dias_previos','contactado_antes','contactos_previos','log_contactos_previos']].head(8).to_string(index=False))

numeric_cols = ['edad','saldo','dia','contactos_campana','dias_previos','contactos_previos','contactado_antes','log_contactos_previos']
cat_cols = ['ocupacion','estado_civil','educacion','mora','hipoteca','prestamo','canal','mes','resultado_previo']

def make_preprocessor(scale_numeric):
    num = StandardScaler() if scale_numeric else 'passthrough'
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([('num', num, numeric_cols),
                             ('cat', cat, cat_cols)], remainder='drop', verbose_feature_names_out=False)

pre_scaled = make_preprocessor(True)
X_preview = pre_scaled.fit_transform(entrenamiento)
print('\\nForma matriz preparada (escalada):', X_preview.shape)
print('N\u00famero de columnas preparadas:', X_preview.shape[1])
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

- `contactado_antes` captura si el cliente ya fue contactado en campa\u00f1as anteriores (el -1 de `dias_previos` indica ausencia de contacto previo). Es un indicador binario con sentido comercial directo.
- `log_contactos_previos` = log(1 + contactos_previos) reduce la asimetr\u00eda de un conteo con muchos ceros y valores altos, estabilizando la varianza y mejorando modelos lineales.

Se reemplaza -1 por 0 en `dias_previos` pero se conserva el indicador, de modo que el modelo distingue "no contactado" de "contactado hace 0 d\u00edas". Las categor\u00edas `unknown` se mantienen como nivel propio. Las variables categ\u00f3ricas se codifican con *one-hot* (`handle_unknown='ignore'` para pliegues de prueba).

**Escalamiento por m\u00e9todo:** se estandarizan las columnas num\u00e9ricas con media y desviaci\u00f3n muestral (ddof=1) para los m\u00e9todos basados en distancias o gradientes (regresi\u00f3n log\u00edstica, SVM, Naive Bayes gaussiano, PCA, K-means y Mahalanobis). Las variables binarias (one-hot) se dejan en 0/1. Los \u00e1rboles, Random Forest y boosting son invariantes a escalamientos mon\u00f3tonos, por lo que pueden usar las columnas sin escalar (aunque el resultado es equivalente). Las transformaciones se aprenden \u00fanicamente con las filas de ajuste de cada pliegue para evitar fuga de informaci\u00f3n.
""")

# Pregunta 2
md("## Pregunta 2. Selecci\u00f3n de variables y modelos supervisados (24 puntos)")

md("""| Modelo | Configuraciones |
|---|---|
| Regresi\u00f3n log\u00edstica | Sin penalizaci\u00f3n: todas las columnas preparadas y una versi\u00f3n con selecci\u00f3n de 10 columnas. |
| \u00c1rbol de decisi\u00f3n | Profundidades m\u00e1ximas 3 y 6; m\u00ednimo 20 observaciones por hoja. |
| Random Forest | 200 \u00e1rboles; ra\u00edz cuadrada del n\u00famero de columnas como candidatas por divisi\u00f3n; m\u00ednimo 10 por hoja. |
| Boosting | Una variante vista en clases; hasta 200 \u00e1rboles y profundidad m\u00e1xima 3. Informen variante y tasa de aprendizaje. |
| SVM radial | C = 1 y gamma = 1/p, donde p es el n\u00famero de columnas preparadas; usen el puntaje de decisi\u00f3n. |
| Naive Bayes | Variante gaussiana sobre las columnas num\u00e9ricas y binarias preparadas. |
""")

md("""### 2(a) (5 puntos)

Comparen regresi\u00f3n log\u00edstica con todas las columnas preparadas y con las 10 de mayor estad\u00edstico F de ANOVA respecto de `contrata`. Informen las columnas seleccionadas en cada pliegue y comparen sus m\u00e9tricas OOF. \u00bfEn qu\u00e9 se diferencia esta selecci\u00f3n de PCA? \u00bfQu\u00e9 limitaci\u00f3n tiene el filtro univariado?
""")

code("""from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

# Asegurar features base
def base_features(df):
    df = df.copy()
    df['contactado_antes'] = (df['dias_previos'] != -1).astype(int)
    df['log_contactos_previos'] = np.log1p(df['contactos_previos'])
    df['dias_previos'] = df['dias_previos'].replace(-1, 0)
    return df

train = base_features(pd.read_csv(CARPETA / 'clientes_entrenamiento.csv'))
y = train['contrata'].to_numpy()
folds = sorted(train['pliegue'].unique())
n = len(train)

numeric_cols = ['edad','saldo','dia','contactos_campana','dias_previos','contactos_previos','contactado_antes','log_contactos_previos']
cat_cols = ['ocupacion','estado_civil','educacion','mora','hipoteca','prestamo','canal','mes','resultado_previo']

def make_preprocessor(scale_numeric):
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    num = StandardScaler() if scale_numeric else 'passthrough'
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([('num', num, numeric_cols), ('cat', cat, cat_cols)], remainder='drop', verbose_feature_names_out=False)

oof_prob = {m: np.full(n, np.nan) for m in ['logreg_all','logreg_10','dt3','dt6','rf','gb','nb']}
oof_score = np.full(n, np.nan)
ref_oof = np.full(n, np.nan)
selected_features_per_fold = {}
train_logloss_trees = {'dt3': [], 'dt6': []}

for fold in folds:
    tr_mask = (train['pliegue'] != fold).to_numpy()
    va_mask = (train['pliegue'] == fold).to_numpy()
    Xtr_df = train.loc[tr_mask]; Xva_df = train.loc[va_mask]
    ytr = y[tr_mask]; yva = y[va_mask]

    # referencia
    ref_oof[va_mask] = ytr.mean()

    # preparacion escalada
    pre_s = make_preprocessor(True)
    Xtr_s = pre_s.fit_transform(Xtr_df)
    Xva_s = pre_s.transform(Xva_df)
    feature_names = list(pre_s.get_feature_names_out())

    # logistica todas las columnas
    lr_all = LogisticRegression(penalty=None, max_iter=10000, random_state=2026)
    lr_all.fit(Xtr_s, ytr)
    oof_prob['logreg_all'][va_mask] = lr_all.predict_proba(Xva_s)[:, 1]

    # seleccion 10 por F de ANOVA
    sel = SelectKBest(f_classif, k=10).fit(Xtr_s, ytr)
    Xtr_10 = sel.transform(Xtr_s); Xva_10 = sel.transform(Xva_s)
    selected = [feature_names[i] for i in sel.get_support(indices=True)]
    selected_features_per_fold[fold] = selected
    lr_10 = LogisticRegression(penalty=None, max_iter=10000, random_state=2026)
    lr_10.fit(Xtr_10, ytr)
    oof_prob['logreg_10'][va_mask] = lr_10.predict_proba(Xva_10)[:, 1]

    # Naive Bayes gaussiano
    nb = GaussianNB()
    nb.fit(Xtr_s, ytr)
    oof_prob['nb'][va_mask] = nb.predict_proba(Xva_s)[:, 1]

    # SVM radial
    p = Xtr_s.shape[1]
    svm = SVC(C=1, gamma=1/p, kernel='rbf')
    svm.fit(Xtr_s, ytr)
    oof_score[va_mask] = svm.decision_function(Xva_s)

    # preparacion sin escalar para arboles
    pre_u = make_preprocessor(False)
    Xtr_u = pre_u.fit_transform(Xtr_df)
    Xva_u = pre_u.transform(Xva_df)

    # arbol profundidad 3
    dt3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=2026)
    dt3.fit(Xtr_u, ytr)
    oof_prob['dt3'][va_mask] = dt3.predict_proba(Xva_u)[:, 1]
    train_logloss_trees['dt3'].append(log_loss(ytr, dt3.predict_proba(Xtr_u)[:, 1]))

    # arbol profundidad 6
    dt6 = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=2026)
    dt6.fit(Xtr_u, ytr)
    oof_prob['dt6'][va_mask] = dt6.predict_proba(Xva_u)[:, 1]
    train_logloss_trees['dt6'].append(log_loss(ytr, dt6.predict_proba(Xtr_u)[:, 1]))

    # random forest
    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=10, random_state=2026, n_jobs=-1)
    rf.fit(Xtr_u, ytr)
    oof_prob['rf'][va_mask] = rf.predict_proba(Xva_u)[:, 1]

    # boosting (GradientBoostingClassifier)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=2026)
    gb.fit(Xtr_u, ytr)
    oof_prob['gb'][va_mask] = gb.predict_proba(Xva_u)[:, 1]

print('Columnas seleccionadas por pliegue (log\u00edstica con 10):')
for fold, cols in selected_features_per_fold.items():
    print(f'  Pliegue {fold}: {cols}')

print('\\nLog loss de entrenamiento medio por \u00e1rbol:')
for m, vals in train_logloss_trees.items():
    print(f'  {m}: {np.mean(vals):.4f}')
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

Las 10 columnas seleccionadas var\u00edan levemente entre pliegues, pero son consistentes: `contactado_antes`, `log_contactos_previos`, `canal_cellular`, `canal_unknown`, `mes_mar`, `mes_oct`, `resultado_previo_success`, `resultado_previo_unknown`, m\u00e1s alguna variable de contactos previos/hipoteca. Esto indica que el historial de contacto previo y el canal/mes de contacto son los predictores univariados m\u00e1s informativos.

**Diferencia con PCA:** el filtro univariado (ANOVA F) selecciona variables originales que mejor separan la respuesta de forma marginal. PCA, en cambio, construye combinaciones lineales ortogonales que maximizan la varianza total, sin usar la respuesta; puede mezclar variables poco relevantes y no selecciona variables individuales. **Limitaci\u00f3n del filtro univariado:** eval\u00faa cada predictor por separado, ignora correlaciones y redundancias entre variables, no captura interacciones y puede seleccionar variables que aportan informaci\u00f3n duplicada. Adem\u00e1s, al ser univariado, no considera el desempe\u00f1o multivariado del modelo final.
""")

md("""### 2(b) (11 puntos)

Comparen las configuraciones de la tabla, incluidas las dos regresiones de 2(a), mediante sus predicciones OOF. Presenten log loss, Brier y AUC; para SVM, solo AUC. Incluyan el modelo de referencia y, para ambos \u00e1rboles, el log loss medio de entrenamiento de los cinco ajustes. Seleccionen el modelo probabil\u00edstico con menor log loss OOF.
""")

code("""rows = []
for m in ['logreg_all','logreg_10','dt3','dt6','rf','gb','nb']:
    rows.append({
        'Modelo': m,
        'Log loss OOF': log_loss(y, oof_prob[m]),
        'Brier OOF': brier_score_loss(y, oof_prob[m]),
        'AUC OOF': roc_auc_score(y, oof_prob[m])
    })
rows.append({
    'Modelo': 'svm (score)',
    'Log loss OOF': np.nan,
    'Brier OOF': np.nan,
    'AUC OOF': roc_auc_score(y, oof_score)
})
rows.append({
    'Modelo': 'referencia',
    'Log loss OOF': log_loss(y, ref_oof),
    'Brier OOF': brier_score_loss(y, ref_oof),
    'AUC OOF': roc_auc_score(y, ref_oof)
})
metrics_df = pd.DataFrame(rows)
print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\\nLog loss entrenamiento medio:')
print(f'  \u00e1rbol prof. 3: {np.mean(train_logloss_trees["dt3"]):.4f}')
print(f'  \u00e1rbol prof. 6: {np.mean(train_logloss_trees["dt6"]):.4f}')
mejor = metrics_df.loc[metrics_df['Log loss OOF'].idxmin(), 'Modelo']
print('\\nModelo probabil\u00edstico con menor log loss OOF:', mejor)
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

El **Random Forest** tiene el menor log loss OOF (0,3096) y tambi\u00e9n el mayor AUC (0,7486) y el menor Brier (0,0891), por lo que se selecciona como modelo probabil\u00edstico. Le siguen el boosting (0,3191) y la regresi\u00f3n log\u00edstica con todas las columnas (0,3150). Los \u00e1rboles individuales y el SVM ordenan peor. La regresi\u00f3n con 10 variables pierde algo de desempe\u00f1o frente a usar todas, pero gana simplicidad. Naive Bayes tiene AUC razonable (0,7138) pero un log loss muy alto (2,5005) por probabilidades extremas (sobreconfianza). El modelo de referencia (probabilidad constante 0,1153) queda claramente por debajo en todas las m\u00e9tricas (log loss 0,3574, AUC 0,5), confirmando que los modelos aprenden se\u00f1al real.
""")

md("""### 2(c) (8 puntos)

Interpreten los resultados considerando: profundidad y sobreajuste de los \u00e1rboles; diferencias entre bagging, Random Forest y boosting; independencia condicional de Naive Bayes y limitaciones de la variante gaussiana para columnas binarias; efecto de C, gamma y el escalamiento en SVM. \u00bfUn mejor ordenamiento de las observaciones implica mejores probabilidades? Fundamenten.
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

- **Profundidad y sobreajuste:** el \u00e1rbol de profundidad 6 tiene menor log loss de entrenamiento (0,3020) que el de profundidad 3 (0,3199), es decir, ajusta m\u00e1s la muestra de entrenamiento, pero su mejora OOF es modesta (0,3230 vs 0,3298). La profundidad 3 es m\u00e1s estable y menos propensa a sobreajuste; la profundidad 6 captura m\u00e1s interacciones a costa de mayor varianza. El m\u00ednimo de 20 observaciones por hoja act\u00faa como regularizaci\u00f3n.
- **Bagging vs Random Forest vs boosting:** bagging reduce varianza promediando \u00e1rboles entrenados en muestras bootstrap. Random Forest a\u00f1ade selecci\u00f3n aleatoria de variables por divisi\u00f3n, descorrelacionando los \u00e1rboles y mejorando la reducci\u00f3n de varianza (mejor resultado aqu\u00ed). Boosting ajusta secuencialmente los errores residuales con \u00e1rboles superficiales (profundidad 3) y tasa de aprendizaje 0,1; reduce sesgo pero es m\u00e1s sensible al ruido y al sobreajuste si no se controla.
- **Naive Bayes:** asume independencia condicional entre predictores dado la clase, supuesto violado por variables correlacionadas (p. ej. `contactos_previos` y `contactado_antes`). La variante gaussiana modela cada columna como normal; las variables binarias one-hot no son gaussianas y generan probabilidades extremas, de ah\u00ed el log loss alto. A\u00fan as\u00ed, el ordenamiento (AUC 0,7138) es aceptable.
- **SVM radial:** C = 1 controla el margen (mayor C = menos regularizaci\u00f3n); gamma = 1/p define el ancho del kernel (mayor gamma = fronteras m\u00e1s complejas y riesgo de sobreajuste). El escalamiento es cr\u00edtico porque el kernel RBF se basa en distancias euclidianas: sin estandarizar, variables con mayor escala (saldo) dominar\u00edan. Con C=1 y gamma=1/p el modelo es conservador y queda por debajo del Random Forest en AUC (0,6743).
- **Mejor ordenamiento \u2260 mejores probabilidades:** AUC mide solo el ordenamiento (ranking), mientras que log loss y Brier miden calibraci\u00f3n. Un modelo puede ordenar bien pero dar probabilidades mal calibradas (Naive Bayes). Para decidir umbrales y costos se necesitan probabilidades bien calibradas, no solo buen ranking.
""")

# Pregunta 3
md("## Pregunta 3. Evaluaci\u00f3n y decisi\u00f3n (10 puntos)")

md("""### 3(a) (4 puntos)

Con las probabilidades OOF del modelo elegido, eval\u00faen los umbrales 0.05; 0.10; ...; 0.95. Predigan contrataci\u00f3n si p es mayor o igual al umbral. Elijan el que minimiza el costo hipot\u00e9tico C(t) = FP(t) + 5 FN(t), donde FP es una alerta de contrataci\u00f3n que no ocurre y FN una contrataci\u00f3n no detectada. Presenten una tabla o curva e interpreten ambos errores.
""")

code("""prob_rf = oof_prob['rf']
rows = []
for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
    pred = (prob_rf >= t).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rows.append({'umbral': t, 'FP': fp, 'FN': fn, 'C': fp + 5*fn, 'C/n': (fp + 5*fn)/len(y)})
thr_df = pd.DataFrame(rows)
print(thr_df.to_string(index=False))
best_umbral = float(thr_df.loc[thr_df['C'].idxmin(), 'umbral'])
print('\\nUmbral que minimiza C(t):', best_umbral, '| C =', int(thr_df['C'].min()))

plt.figure(figsize=(6,4))
plt.plot(thr_df['umbral'], thr_df['C'], marker='o')
plt.xlabel('Umbral'); plt.ylabel('C(t) = FP + 5 FN')
plt.title('Costo por umbral (OOF, Random Forest)')
plt.grid(True); plt.show()

print('\\nMatriz de confusi\u00f3n OOF con umbral elegido:')
pred = (prob_rf >= best_umbral).astype(int)
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

El umbral que minimiza el costo es **0,15** (C = 1490; C/n = 0,412). A ese umbral se obtienen 550 falsos positivos y 188 falsos negativos (TP = 229, TN = 2650).

- **Falso positivo (FP):** se contacta a un cliente que no contratar\u00e1; es un contacto poco productivo (costo 1).
- **Falso negativo (FN):** no se contacta a un cliente que s\u00ed habr\u00eda contratado; es una oportunidad perdida (costo 5).

Como el costo del falso negativo es 5 veces mayor, el umbral \u00f3ptimo (0,15) es mucho m\u00e1s bajo que 0,50: se prefiere contactar a m\u00e1s clientes para no perder oportunidades, aceptando m\u00e1s contactos improductivos. Umbrales muy bajos (0,05) disparan los falsos positivos y el costo total; umbrales altos dejan escapar la mayor\u00eda de los contratos.
""")

md("""### 3(b) (6 puntos)

Reajusten el modelo elegido con todo el entrenamiento y eval\u00faen en `clientes_prueba.csv`. Informen log loss, Brier, AUC y curva ROC. Comparen el umbral elegido con 0.50 mediante matriz de confusi\u00f3n, sensibilidad, especificidad y C/n; incluyan el modelo de referencia. \u00bfQu\u00e9 recomendar\u00edan al banco? \u00bfQu\u00e9 permiten concluir los resultados y qu\u00e9 falta verificar para una campa\u00f1a futura?
""")

code("""prueba = base_features(pd.read_csv(CARPETA / 'clientes_prueba.csv'))
pre_final = make_preprocessor(False)
Xtr_all = pre_final.fit_transform(train)
Xte = pre_final.transform(prueba)
rf_final = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=10, random_state=2026, n_jobs=-1)
rf_final.fit(Xtr_all, train['contrata'].to_numpy())
p_test = rf_final.predict_proba(Xte)[:, 1]
y_test = prueba['contrata'].to_numpy()

print('M\u00e9tricas en prueba (Random Forest):')
print('  Log loss:', round(log_loss(y_test, p_test), 4))
print('  Brier:   ', round(brier_score_loss(y_test, p_test), 4))
print('  AUC:     ', round(roc_auc_score(y_test, p_test), 4))

# Curva ROC
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, p_test)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'Random Forest (AUC={roc_auc_score(y_test, p_test):.3f})')
plt.plot([0,1],[0,1],'k--', label='Azar')
plt.xlabel('1 - Especificidad (FPR)'); plt.ylabel('Sensibilidad (TPR)')
plt.title('Curva ROC - prueba'); plt.legend(); plt.grid(True); plt.show()

# Referencia
ref_prob = train['contrata'].mean()
print('\\nModelo de referencia (probabilidad constante):')
print('  p =', round(ref_prob, 4))
print('  Log loss:', round(log_loss(y_test, np.full(len(y_test), ref_prob)), 4))
print('  Brier:   ', round(brier_score_loss(y_test, np.full(len(y_test), ref_prob)), 4))
print('  AUC:     ', round(roc_auc_score(y_test, np.full(len(y_test), ref_prob)), 4))

print('\\nComparaci\u00f3n de umbrales en prueba:')
for t in [best_umbral, 0.50]:
    pred = (p_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    cost = fp + 5*fn
    print(f'  t={t}: TN={tn} FP={fp} FN={fn} TP={tp} | sens={sens:.4f} spec={spec:.4f} | C={cost} C/n={cost/len(y_test):.4f}')
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

En prueba, el Random Forest obtiene log loss 0,3109, Brier 0,0884 y AUC 0,7370, resultados coherentes con la validaci\u00f3n cruzada (sin sobreajuste evidente). El modelo de referencia (probabilidad constante 0,1153) obtiene log loss 0,3569, Brier 0,1018 y AUC 0,5, por lo que el modelo aporta valor real.

Comparando umbrales en prueba:
- **Umbral 0,15 (elegido):** sensibilidad 0,567, especificidad 0,805, C = 381 (C/n = 0,421).
- **Umbral 0,50:** sensibilidad 0,144, especificidad 0,995, C = 449 (C/n = 0,497).

El umbral 0,15 reduce el costo total al capturar m\u00e1s contrataciones, aunque genera m\u00e1s contactos improductivos; el umbral 0,50 es demasiado conservador y deja escapar la mayor\u00eda de las oportunidades.

**Recomendaci\u00f3n al banco:** usar el Random Forest con umbral 0,15 para priorizar la lista de contactos de la campa\u00f1a; contactar primero a los clientes con mayor probabilidad predicha. Los resultados permiten concluir que el modelo discrimina mejor que el azar y que la informaci\u00f3n previa (contactos anteriores, canal, mes) es \u00fatil. Para una campa\u00f1a futura falta verificar: estabilidad temporal de los patrones, calibraci\u00f3n de las probabilidades en la nueva poblaci\u00f3n, validez del supuesto de costo (FP + 5 FN), efecto del contacto sobre la contrataci\u00f3n (causalidad), y monitoreo de deriva de datos y desempe\u00f1o.
""")

# Pregunta 4
md("## Pregunta 4. Componentes principales y agrupamiento (8 puntos)")

md("""### 4(a) (3 puntos)

Realicen PCA. Informen la varianza explicada por PC1 y PC2, su suma y las cargas; grafiquen las observaciones en ese plano. \u00bfQu\u00e9 informaci\u00f3n resume la proyecci\u00f3n y cu\u00e1l omite?
""")

code("""from sklearn.decomposition import PCA

vars4 = ['edad','saldo','contactos_campana','contactos_previos']
X4 = train[vars4].to_numpy(dtype=float)
mu4 = X4.mean(axis=0)
sd4 = X4.std(axis=0, ddof=1)
X4s = (X4 - mu4) / sd4

pca = PCA(n_components=2)
P = pca.fit_transform(X4s)
print('Varianza explicada PC1:', round(pca.explained_variance_ratio_[0], 4))
print('Varianza explicada PC2:', round(pca.explained_variance_ratio_[1], 4))
print('Suma PC1+PC2:', round(pca.explained_variance_ratio_.sum(), 4))
loadings = pd.DataFrame(pca.components_, columns=vars4, index=['PC1','PC2'])
print('\\nCargas:')
print(loadings.round(4).to_string())

plt.figure(figsize=(7,5))
sc = plt.scatter(P[:,0], P[:,1], c=train['contrata'], cmap='coolwarm', s=12, alpha=0.6)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('Observaciones en el plano PC1-PC2')
plt.colorbar(sc, label='contrata'); plt.grid(True); plt.show()
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

PC1 explica el 27,70 % de la varianza y PC2 el 26,67 %, sumando 54,37 %. Las cargas muestran que PC1 combina principalmente edad y saldo (positivos) frente a contactos de campa\u00f1a (negativo), mientras PC2 contrasta contactos previos (positivo) con contactos de campa\u00f1a (negativo). La proyecci\u00f3n resume la estructura lineal dominante de las cuatro variables (nivel socioecon\u00f3mico y actividad de contacto), pero omite el 45,63 % restante de la varianza, relaciones no lineales, interacciones y toda la informaci\u00f3n categ\u00f3rica. Adem\u00e1s, al ser no supervisado, no garantiza separaci\u00f3n de las clases.
""")

md("""### 4(b) (5 puntos)

Ajusten K-means sobre las cuatro variables estandarizadas con k = 2, 3 y 4 y al menos 30 inicializaciones. Comparen WSS y silueta media y elijan el k de mayor silueta. Informen tama\u00f1os, describan perfiles y representen los grupos en el plano PCA. Calculen la proporci\u00f3n de contrataci\u00f3n dentro de cada grupo y expliquen por qu\u00e9 los grupos no equivalen a clases predichas.
""")

code("""from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=30, random_state=2026)
    lab = km.fit_predict(X4s)
    sil = silhouette_score(X4s, lab)
    print(f'k={k}: WSS={km.inertia_:.2f} | silueta={sil:.4f} | tama\u00f1os={np.bincount(lab)}')

# elegir k de mayor silueta
best_k = max([2,3,4], key=lambda k: silhouette_score(X4s, KMeans(n_clusters=k, n_init=30, random_state=2026).fit_predict(X4s)))
km_final = KMeans(n_clusters=best_k, n_init=30, random_state=2026)
clusters = km_final.fit_predict(X4s)
print('\\nK elegido:', best_k)

profile = train[vars4].copy()
profile['cluster'] = clusters
print('\\nPerfil medio por cluster:')
print(profile.groupby('cluster').mean().round(2).to_string())
print('\\nTama\u00f1os:', np.bincount(clusters))
print('\\nProporci\u00f3n de contrataci\u00f3n por cluster:')
print(train.assign(cluster=clusters).groupby('cluster')['contrata'].mean().round(4).to_string())

plt.figure(figsize=(7,5))
plt.scatter(P[:,0], P[:,1], c=clusters, cmap='tab10', s=12, alpha=0.7)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title(f'Clusters K-means (k={best_k}) en el plano PCA')
plt.colorbar(label='cluster'); plt.grid(True); plt.show()
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

El k de mayor silueta es 4 (silueta 0,3631; WSS 7665,01). Los grupos tienen tama\u00f1os 177, 2004, 1289 y 147, con perfiles diferenciados:

- **Cluster 0** (n=177): edad ~40, saldo alto (~1651), pocos contactos de campa\u00f1a (~2,1) pero muchos contactos previos (~6,6). Es el grupo con mayor conversi\u00f3n (26,6 %).
- **Cluster 1** (n=2004): j\u00f3venes (~34), saldo bajo (~931), pocos contactos previos. Conversi\u00f3n 10,8 %.
- **Cluster 2** (n=1289): mayores (~52), saldo alto (~2146), pocos contactos previos. Conversi\u00f3n 11,4 %.
- **Cluster 3** (n=147): edad ~40, saldo medio (~1153), pero con much\u00edsimos contactos de campa\u00f1a (~14,5) y casi sin contactos previos. Es el grupo de menor conversi\u00f3n (4,8 %).

Los grupos **no equivalen a clases predichas** porque K-means es no supervisado: no utiliza `contrata`. Agrupa por similitud en las cuatro variables continuas, de modo que cada cluster contiene mezcla de contrataciones y no contrataciones. La proporci\u00f3n de contrataci\u00f3n var\u00eda entre grupos, pero un cluster no es una predicci\u00f3n individual; adem\u00e1s, la conversi\u00f3n depende tambi\u00e9n de variables categ\u00f3ricas no incluidas en el agrupamiento.
""")

# Pregunta 5
md("## Pregunta 5. Detecci\u00f3n de anomal\u00edas (6 puntos)")

md("""### 5(a) (3 puntos)

Calculen la distancia de Mahalanobis al cuadrado y usen como corte el cuantil 0.99 de una chi-cuadrado con 4 grados de libertad. Informen el corte, la cantidad y el porcentaje de registros marcados. Identifiquen los dos de mayor distancia y presenten sus valores originales.
""")

code("""from scipy.stats import chi2

cov4 = np.cov(X4s, rowvar=False, ddof=1)
cov4_inv = np.linalg.inv(cov4)
# media cero por estandarizacion muestral
d2 = np.einsum('ij,jk,ik->i', X4s, cov4_inv, X4s)
cutoff = chi2.ppf(0.99, df=4)
marked = d2 > cutoff
print('Corte (chi2 0.99, gl=4):', round(float(cutoff), 4))
print('Registros marcados:', int(marked.sum()))
print('Porcentaje marcado:', round(100*marked.mean(), 4), '%')

order = np.argsort(d2)[::-1]
print('\\nDos registros de mayor distancia:')
for rank, idx in enumerate(order[:2], start=1):
    print(f'\\nRank {rank} - indice {idx} - d2 = {d2[idx]:.2f}')
    print(train.iloc[idx].to_string())
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

El corte es 13,2767 (cuantil 0,99 de \u03c7\u00b2 con 4 gl). Se marcan 189 registros, el 5,23 % de la muestra, no el 1 % esperado. Los dos m\u00e1s inusuales son:

1. Registro de edad 60, ocupaci\u00f3n retirado, saldo 71.188 euros, 1 contacto de campa\u00f1a, sin contactos previos (d\u00b2 \u2248 518,62). Es un valor extremo de saldo.
2. Registro de edad 35, ocupaci\u00f3n blue-collar, saldo 280 euros, 50 contactos de campa\u00f1a, sin contactos previos (d\u00b2 \u2248 224,27). Es un valor extremo en n\u00famero de contactos de campa\u00f1a.

Ambos son at\u00edpicos por una variable con valor muy alejado del resto, no necesariamente por error.
""")

md("""### 5(b) (3 puntos)

\u00bfPor qu\u00e9 son inusuales los dos registros identificados? \u00bfEsto demuestra que contienen errores? \u00bfEl corte utilizado garantiza marcar al 1% de la muestra? Fundamenten e indiquen qu\u00e9 revisar\u00edan antes de eliminar un registro.
""")

md("""**Respuesta e interpretaci\u00f3n del grupo:**

Los dos registros son inusuales porque una de sus variables (saldo en el primero, contactos de campa\u00f1a en el segundo) toma un valor muy alejado de la nube de datos; la distancia de Mahalanobis combina desviaciones en todas las variables considerando su covarianza, por lo que un valor extremo en una variable con baja varianza relativa genera una distancia grande.

Esto **no demuestra que contengan errores**: un saldo alto puede corresponder a un cliente con alto patrimonio y 50 contactos puede ser un cliente reiteradamente contactado. La anomal\u00eda estad\u00edstica no implica error de captura.

El corte **no garantiza marcar el 1 % de la muestra**. El cuantil 0,99 de \u03c7\u00b2 con 4 gl supone que las distancias siguen esa distribuci\u00f3n, lo cual requiere normalidad multivariada y que la media y covarianza sean los par\u00e1metros poblacionales. En datos reales, asimetr\u00eda y colas pesadas hacen que la proporci\u00f3n emp\u00edrica difiera (aqu\u00ed 5,23 %). Adem\u00e1s, la covarianza muestral se estima de los mismos datos.

Antes de eliminar un registro revisar\u00eda: verificaci\u00f3n de la fuente/captura, plausibilidad de dominio, consistencia con otras variables, influencia en los modelos (apalancamiento), y alternativas robustas (winsorizaci\u00f3n, transformaci\u00f3n log, modelos robustos) en lugar de eliminaci\u00f3n autom\u00e1tica.
""")

md("""## Criterios de evaluaci\u00f3n

Se evaluar\u00e1n procedimiento, resultados e interpretaci\u00f3n. Se asignar\u00e1 cr\u00e9dito parcial sin descontar dos veces un mismo error arrastrado. Se aceptan diferencias num\u00e9ricas y ganadores distintos cuando la implementaci\u00f3n y la regla de selecci\u00f3n est\u00e9n justificadas.
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"}
}

out_path = Path(r'C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Clase 42 Taller #3\Taller03_Python.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('notebook written:', out_path)