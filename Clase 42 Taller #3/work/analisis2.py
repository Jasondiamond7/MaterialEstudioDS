import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, roc_curve, confusion_matrix
import json

np.random.seed(2026)
CARPETA = Path(r'C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Clase 42 Taller #3')
train = pd.read_csv(CARPETA / 'clientes_entrenamiento.csv')
prueba = pd.read_csv(CARPETA / 'clientes_prueba.csv')

numeric_cols = ['edad','saldo','dia','contactos_campana','dias_previos','contactos_previos','contactado_antes','log_contactos_previos']
cat_cols = ['ocupacion','estado_civil','educacion','mora','hipoteca','prestamo','canal','mes','resultado_previo']

def base_features(df):
    df = df.copy()
    df['contactado_antes'] = (df['dias_previos'] != -1).astype(int)
    df['log_contactos_previos'] = np.log1p(df['contactos_previos'])
    df['dias_previos'] = df['dias_previos'].replace(-1, 0)
    return df

train = base_features(train)
prueba = base_features(prueba)

def make_preprocessor(scale_numeric):
    num = StandardScaler() if scale_numeric else 'passthrough'
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([('num', num, numeric_cols), ('cat', cat, cat_cols)], remainder='drop', verbose_feature_names_out=False)

# OOF RF probs already in oof_predictions.csv
oof = pd.read_csv(CARPETA/'work'/'oof_predictions.csv')
y = oof['y'].to_numpy()
prob = oof['rf'].to_numpy()

# thresholds
rows=[]
for t in np.arange(0.05,1.0,0.05):
    t=round(t,2)
    pred = (prob>=t).astype(int)
    fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum())
    cost = fp + 5*fn
    rows.append({'umbral':t,'FP':fp,'FN':fn,'C':cost,'C/n':cost/len(y)})
thr_df = pd.DataFrame(rows)
print(thr_df.to_string(index=False))
best = thr_df.loc[thr_df['C'].idxmin()]
print('best threshold:', best['umbral'], 'C=',best['C'])

# test evaluation
pre = make_preprocessor(False)  # RF no scaling needed
Xtr = pre.fit_transform(train)
Xte = pre.transform(prueba)
rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=10, random_state=2026, n_jobs=-1)
rf.fit(Xtr, train['contrata'].to_numpy())
pte = rf.predict_proba(Xte)[:,1]
yte = prueba['contrata'].to_numpy()
print('test logloss', log_loss(yte, pte))
print('test brier', brier_score_loss(yte, pte))
print('test auc', roc_auc_score(yte, pte))
# ROC curve points
fpr,tpr,thr = roc_curve(yte, pte)
print('ROC first 5:', list(zip(fpr[:5],tpr[:5])))

# reference on test: train proportion
refp = train['contrata'].mean()
print('ref prob', refp)
print('ref test logloss', log_loss(yte, np.full(len(yte), refp)))
print('ref test brier', brier_score_loss(yte, np.full(len(yte), refp)))
print('ref test auc', roc_auc_score(yte, np.full(len(yte), refp)))

# threshold comparison on test
for t in [best['umbral'], 0.5]:
    pred = (pte>=t).astype(int)
    tn,fp,fn,tp = confusion_matrix(yte,pred).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    cost = fp + 5*fn
    print(f't={t}: CM tn={tn} fp={fp} fn={fn} tp={tp} sens={sens:.4f} spec={spec:.4f} C={cost} C/n={cost/len(yte):.4f}')

# chosen threshold details on OOF
t=best['umbral']
pred=(prob>=t).astype(int)
tn,fp,fn,tp = confusion_matrix(y,pred).ravel()
print('OOF chosen CM', tn,fp,fn,tp)

thr_df.to_csv(CARPETA/'work'/'thresholds.csv', index=False)
np.save(CARPETA/'work'/'test_pred.npy', pte)
import json
json.dump({'threshold':float(best['umbral']),'cost':int(best['C'])}, open(CARPETA/'work'/'chosen.json','w'))
print('saved')