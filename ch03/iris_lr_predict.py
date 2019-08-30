from sklearn.linear_model import LogisticRegression

from iris_split import y_train
from iris_std import X_train_std, X_test_std

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

proba = lr.predict_proba(X_test_std[:3, :])
print('predict_proba: ', proba)

sum = lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
print('predict_proba.sum: ', sum)

argmax = lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
print('predict_proba.argmax: ', argmax)

predict = lr.predict(X_test_std[:3, :])
print('predict: ', predict)

predict = lr.predict(X_test_std[0, :].reshape(1, -1))
print('predict([3]->[1,-1]): ', predict)