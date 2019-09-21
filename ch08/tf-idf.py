import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from raw_tf import docs, count

np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())

