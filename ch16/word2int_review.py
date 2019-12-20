import pyprind
import pandas as pd
from string import punctuation
import re
import numpy as np
from collections import Counter


df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))



## Preprocessing the data:
## Separate words and 
## count each word's occurrence

counts = Counter()
pbar = pyprind.ProgBar(len(df['review']),
                       title='Counting words occurences')
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' ' 
                        for c in review]).lower()
    df.loc[i,'review'] = text
    pbar.update()
    counts.update(text.split())

## Create a mapping:
## Map each unique word to an integer

word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}


mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']),
                       title='Map reviews to ints')
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()

