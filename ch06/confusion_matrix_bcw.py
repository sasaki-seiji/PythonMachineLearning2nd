from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from prepare_breast_cancer import X_train, y_train, X_test, y_test

# 2020.01.04 add
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
