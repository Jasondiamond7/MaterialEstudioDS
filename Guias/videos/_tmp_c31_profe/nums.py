import sys
sys.path.insert(0, r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe")
from club import *
print(len(df), df.cancelo.mean(), yte.mean(), len(yte), len(ytr))
print(confusion_matrix(yte,pred).tolist(), accuracy_score(yte,pred), precision_score(yte,pred), recall_score(yte,pred), f1_score(yte,pred), roc_auc_score(yte,proba))
print("sin cw", precision_score(yte,clf_sin.predict(Xte)), recall_score(yte,clf_sin.predict(Xte)))
print("base", 1-yte.mean())
lr = clf.named_steps['lr']; print(dict(zip(FEATS_CLF, lr.coef_[0].round(2))), lr.intercept_.round(2))
print(df.describe().round(1).to_string())
