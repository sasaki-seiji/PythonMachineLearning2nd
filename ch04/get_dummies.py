
import pandas as pd

from map_class import df

# one-hot encoding via pandas

pd.get_dummies(df[['price', 'color', 'size']])
# 2019.09.09 add
if __name__ == '__main__':
    print(pd.get_dummies(df[['price', 'color', 'size']]))


# multicollinearity guard in get_dummies

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
if __name__ == '__main__':
    print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

