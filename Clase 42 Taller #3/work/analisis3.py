import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

np.random.seed(2026)
CARPETA = Path(r'C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Clase 42 Taller #3')
train = pd.read_csv(CARPETA / 'clientes_entrenamiento.csv')
vars4 = ['edad','saldo','contactos_campana','contactos_previos']
X = train[vars4].to_numpy(dtype=float)
scaler = StandardScaler()  # sample std ddof=1? sklearn uses ddof=0 (population) by default!
# Requirement: estandarizar con media y desviacion muestral (ddof=1). Do manually.
mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)
Xs = (X - mu) / sd

# PCA
pca = PCA(n_components=2)
P = pca.fit_transform(Xs)
print('explained variance:', pca.explained_variance_ratio_)
print('sum:', pca.explained_variance_ratio_.sum())
print('loadings (components x vars):')
print(pd.DataFrame(pca.components_, columns=vars4, index=['PC1','PC2']))
print('means original:', mu)
print('std sample:', sd)

# KMeans
for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=30, random_state=2026)
    lab = km.fit_predict(Xs)
    sil = silhouette_score(Xs, lab)
    print(f'k={k}: WSS={km.inertia_:.2f} silhouette={sil:.4f} sizes={np.bincount(lab)}')

# choose best silhouette
best_k=None; best_sil=-1
for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=30, random_state=2026)
    lab = km.fit_predict(Xs)
    sil = silhouette_score(Xs, lab)
    if sil>best_sil:
        best_sil=sil; best_k=k
print('best k', best_k, best_sil)
km = KMeans(n_clusters=best_k, n_init=30, random_state=2026)
labels = km.fit_predict(Xs)
# profiles
prof = train[vars4].copy()
prof['cluster']=labels
print(prof.groupby('cluster').mean())
print('sizes', np.bincount(labels))
print('contrata proportion by cluster:')
print(train.assign(cluster=labels).groupby('cluster')['contrata'].mean())
# save
np.save(CARPETA/'work'/'pca.npy', P)
np.save(CARPETA/'work'/'clusters.npy', labels)
np.save(CARPETA/'work'/'Xs4.npy', Xs)
pd.DataFrame({'PC1':P[:,0],'PC2':P[:,1],'cluster':labels}).to_csv(CARPETA/'work'/'pca_clusters.csv',index=False)

# Mahalanobis
cov = np.cov(Xs, rowvar=False, ddof=1)
cov_inv = np.linalg.inv(cov)
d2 = np.array([ (x @ cov_inv @ x) for x in Xs])  # since mean 0
cutoff = chi2.ppf(0.99, df=4)
print('cutoff', cutoff)
marked = d2 > cutoff
print('marked count', marked.sum(), 'pct', marked.mean()*100)
order = np.argsort(d2)[::-1]
print('top5 indices', order[:5])
print('top5 d2', d2[order[:5]])
for idx in order[:2]:
    print('idx', idx)
    print(train.iloc[idx][vars4].to_dict())
    print('d2', d2[idx])
np.save(CARPETA/'work'/'mahal.npy', d2)
pd.DataFrame({'d2':d2}).to_csv(CARPETA/'work'/'mahal.csv',index=False)
print('saved')