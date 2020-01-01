import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ## Loading the Breast Cancer Wisconsin dataset




df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

# if the Breast Cancer dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df = pd.read_csv('wdbc.data', header=None), 

df.head()
# 2020.01.01 add
print('head: ', df.head())




df.shape
# 2020.01.01 add
print('shape: ', df.shape)





X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
# 2020.01.01 add
print('classes_: ', le.classes_)



le.transform(['M', 'B'])
# 2020.01.01
print('transform: ', le.transform(['M', 'B']))




X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)
