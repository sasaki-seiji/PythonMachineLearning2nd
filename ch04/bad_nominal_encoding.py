from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from map_class import df

# ## Performing one-hot encoding on nominal features



X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X
# 2019.09.08 add
if __name__ == '__main__':
    print(X)

