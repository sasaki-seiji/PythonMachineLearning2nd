import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# # Partitioning a dataset into a seperate training and test set



df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# 2019.09.10 change
#print('Class labels', np.unique(df_wine['Class label']))
if __name__ == '__main__':
    print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()
# 2019.09.09 add
if __name__ == '__main__':
    print(df_wine.head())




X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

# 2019.09.09 add
if __name__ == '__main__':
    print(X_test[:2, :])
    print(y_test[:2])