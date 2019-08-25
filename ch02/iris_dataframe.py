import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
tail = df.tail()
if __name__ == '__main__':
    print(tail)