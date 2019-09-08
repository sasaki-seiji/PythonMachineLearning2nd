from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import ColumnTransformer


from map_class import df

# ## Performing one-hot encoding on nominal features



X = df[['color', 'size', 'price']].values
# 2019.09.09 add
if __name__ == '__main__':
    print(X)

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X
# 2019.09.08 add
if __name__ == '__main__':
    print(X)

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
# 2019.09.09 add
if __name__ == '__main__':
    print(ohe.fit_transform(X).toarray())




# return dense array so that we can skip
# the toarray step

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
ohe.fit_transform(X)
# 2019.09.09 add
if __name__ == '__main__':
    print(ohe.fit_transform(X))

# multicollinearity guard for the OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()[:, 1:]
if __name__ == '__main__':
    print(ohe.fit_transform(X).toarray()[:, 1:])
