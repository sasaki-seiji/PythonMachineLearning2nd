import pandas as pd
from io import StringIO
import sys

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))

if __name__ == '__main__':
    # 2019.09.08 change
    #df
    print(df)


    # 2019.09.08 change
    #df.isnull().sum()
    print(df.isnull().sum())




    # access the underlying NumPy array
    # via the `values` attribute
    # 2019.09.08 change
    #df.values
    print(df.values)
