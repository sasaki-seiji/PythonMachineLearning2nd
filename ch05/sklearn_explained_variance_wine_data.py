from sklearn.decomposition import PCA

from std_wine_data import X_train_std

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
# 2019.12.28 add
print(pca.explained_variance_ratio_)
