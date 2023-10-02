import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score

wine = load_wine()
print(wine.DESCR)
print(wine.feature_names)
print(wine.target_names)
print(wine.target)

D_wine = wine.data
target = wine.target
ac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='single')
ac.fit(D_wine)
print(ac.labels_)
cm = contingency_matrix(ac.labels_, target)
print(cm)

sil = silhouette_score(D_wine, ac.labels_)
print(sil)
f1 = f1_score(target, ac.labels_, average='weighted')
print(f1)