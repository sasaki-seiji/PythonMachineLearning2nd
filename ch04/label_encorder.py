
from sklearn.preprocessing import LabelEncoder

from categorical_data import df

# Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y
# 2019.09.08 add
if __name__ == '__main__':
    print(y)



# reverse mapping
class_le.inverse_transform(y)
# 2019.09.08 add
if __name__ == '__main__':
    print(class_le.inverse_transform(y))

# Note: The deprecation warning shown above is due to an implementation detail in scikit-learn. It was already addressed in a pull request (https://github.com/scikit-learn/scikit-learn/pull/9816), and the patch will be released with the next version of scikit-learn (i.e., v. 0.20.0).
