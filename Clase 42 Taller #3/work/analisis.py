import numpy as np
import pandas as pd
from pathlib import Path
import sys
np.random.seed(2026)
CARPETA = Path(r'C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Clase 42 Taller #3')
train = pd.read_csv(CARPETA / 'clientes_entrenamiento.csv')

numeric_cols = ['edad','saldo','dia','contactos_campana','dias_previos','contactos_previos','contactado_antes','log_contactos_previos']
cat_cols = ['ocupacion','estado_civil','educacion','mora','hipoteca','prestamo','canal','mes','resultado_previo']

def base_features(df):
    df = df.copy()
    df['contactado_antes'] = (df['dias_previos'] != -1).astype(int)
    df['log_contactos_previos'] = np.log1p(df['contactos_previos'])
    df['dias_previos'] = df['dias_previos'].replace(-1, 0)
    return df

train = base_features(train)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

def make_preprocessor(scale_numeric):
    num = StandardScaler() if scale_numeric else 'passthrough'
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([('num', num, numeric_cols), ('cat', cat, cat_cols)], remainder='drop', verbose_feature_names_out=False)

folds = sorted(train['pliegue'].unique())
n = len(train)
y = train['contrata'].to_numpy()

# OOF containers
models = ['logreg_all','logreg_10','dt3','dt6','rf','gb','nb']
oof_prob = {m: np.full(n, np.nan) for m in models}
oof_score = np.full(n, np.nan)  # SVM decision
selected_features_per_fold = {}
train_logloss_trees = {'dt3':[], 'dt6':[]}
ref_oof = np.full(n, np.nan)

for fold in folds:
    tr = train['pliegue'] != fold
    va = train['pliegue'] == fold
    Xtr_df = train.loc[tr]; Xva_df = train.loc[va]
    ytr = y[tr.to_numpy()]; yva = y[va.to_numpy()]
    tr_idx = tr.to_numpy(); va_idx = va.to_numpy()

    # reference
    ref_prob = ytr.mean()
    ref_oof[va_idx] = ref_prob

    # --- logistic / selection with scaling
    pre_scaled = make_preprocessor(True)
    Xtr_s = pre_scaled.fit_transform(Xtr_df)
    Xva_s = pre_scaled.transform(Xva_df)
    feature_names = list(pre_scaled.get_feature_names_out())

    # logreg all
    lr = LogisticRegression(penalty=None, max_iter=10000, random_state=2026)
    lr.fit(Xtr_s, ytr)
    oof_prob['logreg_all'][va_idx] = lr.predict_proba(Xva_s)[:,1]

    # logreg 10
    sel = SelectKBest(f_classif, k=10).fit(Xtr_s, ytr)
    Xtr_10 = sel.transform(Xtr_s); Xva_10 = sel.transform(Xva_s)
    selected = [feature_names[i] for i in sel.get_support(indices=True)]
    selected_features_per_fold[fold] = selected
    lr10 = LogisticRegression(penalty=None, max_iter=10000, random_state=2026)
    lr10.fit(Xtr_10, ytr)
    oof_prob['logreg_10'][va_idx] = lr10.predict_proba(Xva_10)[:,1]

    # NB on scaled prepared columns
    nb = GaussianNB()
    nb.fit(Xtr_s, ytr)
    oof_prob['nb'][va_idx] = nb.predict_proba(Xva_s)[:,1]

    # SVM radial
    p = Xtr_s.shape[1]
    svm = SVC(C=1, gamma=1/p, kernel='rbf')
    svm.fit(Xtr_s, ytr)
    oof_score[va_idx] = svm.decision_function(Xva_s)

    # unscaled preprocessor for trees
    pre_unscaled = make_preprocessor(False)
    Xtr_u = pre_unscaled.fit_transform(Xtr_df)
    Xva_u = pre_unscaled.transform(Xva_df)

    # dt3
    dt3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=2026)
    dt3.fit(Xtr_u, ytr)
    oof_prob['dt3'][va_idx] = dt3.predict_proba(Xva_u)[:,1]
    # train logloss
    tr_prob = dt3.predict_proba(Xtr_u)[:,1]
    train_logloss_trees['dt3'].append(log_loss(ytr, tr_prob))

    # dt6
    dt6 = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=2026)
    dt6.fit(Xtr_u, ytr)
    oof_prob['dt6'][va_idx] = dt6.predict_proba(Xva_u)[:,1]
    tr_prob = dt6.predict_proba(Xtr_u)[:,1]
    train_logloss_trees['dt6'].append(log_loss(ytr, tr_prob))

    # rf
    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=10, random_state=2026, n_jobs=-1)
    rf.fit(Xtr_u, ytr)
    oof_prob['rf'][va_idx] = rf.predict_proba(Xva_u)[:,1]

    # boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=2026)
    gb.fit(Xtr_u, ytr)
    oof_prob['gb'][va_idx] = gb.predict_proba(Xva_u)[:,1]

print('Selected features per fold:')
for f,v in selected_features_per_fold.items():
    print(f, v)

print('\nTrain logloss trees:')
for m,v in train_logloss_trees.items():
    print(m, v, 'mean', np.mean(v))

def metrics_prob(y, prob):
    return log_loss(y, prob), brier_score_loss(y, prob), roc_auc_score(y, prob)

print('\nOOF metrics:')
for m in models:
    ll, br, auc = metrics_prob(y, oof_prob[m])
    print(f'{m}: logloss={ll:.4f} brier={br:.4f} auc={auc:.4f}')
# reference
ll, br, auc = metrics_prob(y, ref_oof)
print(f'ref: logloss={ll:.4f} brier={br:.4f} auc={auc:.4f}')
# svm auc
print('svm auc:', roc_auc_score(y, oof_score))

# save oof
out = pd.DataFrame({'y':y})
for m in models: out[m]=oof_prob[m]
out['svm_score']=oof_score
out['ref']=ref_oof
out.to_csv(CARPETA/'work'/'oof_predictions.csv', index=False)
import json
json.dump({str(k):v for k,v in selected_features_per_fold.items()}, open(CARPETA/'work'/'selected.json','w'))
print('saved')