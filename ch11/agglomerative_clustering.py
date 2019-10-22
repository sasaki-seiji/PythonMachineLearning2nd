# ## Applying agglomerative clustering via scikit-learn
from sklearn.cluster import AgglomerativeClustering

# 2019.10.22 add
from random_sample_5x3 import X



ac = AgglomerativeClustering(n_clusters=3, 
                             affinity='euclidean', 
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)

ac = AgglomerativeClustering(n_clusters=2, 
                             affinity='euclidean', 
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
