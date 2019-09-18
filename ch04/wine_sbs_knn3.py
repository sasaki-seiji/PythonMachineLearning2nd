
from wine_dataset import df_wine, y_train, y_test
from wine_std import X_train_std, X_test_std
from wine_sbs_knn import sbs, knn

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])




knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))




knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
