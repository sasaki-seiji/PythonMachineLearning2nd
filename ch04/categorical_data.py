import pandas as pd

# # Handling categorical data

# ## Nominal and ordinal features




df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
df
# 2019.09.08 add
if __name__ == '__main__':
    print(df)
