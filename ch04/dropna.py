from pandas_csv import df

# 2019.09.08 add
print(df)

# remove rows that contain missing values

# 2019.09.08 change
#df.dropna(axis=0)
print(df.dropna(axis=0))




# remove columns that contain missing values
# 2019.09.08 change
#df.dropna(axis=1)
print(df.dropna(axis=1))




# remove columns that contain missing values
# 2019.09.08 remove: duplicated
#df.dropna(axis=1)




# only drop rows where all columns are NaN
# 2019.09.08 change
#df.dropna(how='all')  
print(df.dropna(how='all')  )




# drop rows that have less than 3 real values 
# 2019.09.08 change
#df.dropna(thresh=4)
print(df.dropna(thresh=4))




# only drop rows where NaN appear in specific columns (here: 'C')
# 2019.09.08 change
#df.dropna(subset=['C'])
print(df.dropna(subset=['C']))
