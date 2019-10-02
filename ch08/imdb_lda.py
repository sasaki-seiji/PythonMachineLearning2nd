import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ## Topic modeling

# ### Decomposing text documents with Latent Dirichlet Allocation

# ### Latent Dirichlet Allocation with scikit-learn




df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)




## @Readers: PLEASE IGNORE THIS CELL
##
## This cell is meant to create a smaller dataset if
## the notebook is run on the Travis Continuous Integration
## platform to test the code on a smaller dataset
## to prevent timeout errors and just serves a debugging tool
## for this notebook

if 'TRAVIS' in os.environ:
    df.loc[:500].to_csv('movie_data.csv')
    df = pd.read_csv('movie_data.csv', nrows=500)
    print('SMALL DATA SUBSET CREATED FOR TESTING')





count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)




""" 2019.10.02 change
lda = LatentDirichletAllocation(n_topics=10,
                                random_state=123,
                                learning_method='batch')
"""
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)




lda.components_.shape




n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))


# Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
#     
# 1. Generally bad movies (not really a topic category)
# 2. Movies about families
# 3. War movies
# 4. Art movies
# 5. Crime movies
# 6. Horror movies
# 7. Comedies
# 8. Movies somehow related to TV shows
# 9. Movies based on books
# 10. Action movies

# To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category (category 6 at index position 5):



horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')
