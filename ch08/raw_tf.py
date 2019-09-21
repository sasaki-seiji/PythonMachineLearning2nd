import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)


# Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:


# 2019.09.21 change
#print(count.vocabulary_)
#print(bag.toarray())
if __name__ == '__main__':
    print(count.vocabulary_)
    print(bag.toarray())
