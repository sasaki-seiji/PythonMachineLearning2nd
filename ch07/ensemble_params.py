from sklearn.model_selection import GridSearchCV

from rocs_majority_voting_iris import mv_clf, X_train, y_train


mv_clf.get_params()
# 2020.01.07 add
print(mv_clf.get_params())




params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_['mean_test_score'][r], 
             grid.cv_results_['std_test_score'][r] / 2.0, 
             grid.cv_results_['params'][r]))




print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

# 2020.01.07 add
print(mv_clf.get_params())
