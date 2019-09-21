import pandas as pd




df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
# 2019.09.21 add
if __name__ == '__main__':
    print(df.head(3))