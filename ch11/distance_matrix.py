import pandas as pd
from scipy.spatial.distance import pdist, squareform

from random_sample_5x3 import labels, df

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)
row_dist

# 2019.10.22 add
if __name__ == '__main__':
    print(row_dist)