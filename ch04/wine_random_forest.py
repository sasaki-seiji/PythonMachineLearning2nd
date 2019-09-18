# # Assessing feature importance with Random Forests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from wine_dataset import df_wine, X_train, y_train



feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# 2019.09.18 change
if __name__ == '__main__':
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                feat_labels[indices[f]], 
                                importances[indices[f]]))

    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            align='center')

    plt.xticks(range(X_train.shape[1]), 
            feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('images/04_09.png', dpi=300)
    plt.show()
