from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from housing_all_slr import y_train, y_train_pred, y_test, y_test_pred

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

