from sklearn.preprocessing import Imputer

from pandas_csv import df

# 2019.09.08 add : print original data
print(df)

# impute missing values via the column mean


imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
# 2019.09.08 change
#imputed_data
print(imputed_data)
