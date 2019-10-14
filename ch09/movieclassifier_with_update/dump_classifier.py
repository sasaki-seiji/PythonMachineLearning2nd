import os
import pickle
#import pyprind
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

from vectorizer import tokenizer, stop

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def dump_classifier(clf_path):
    vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

    if Version(sklearn_version) < '0.18':
        clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    else:
        clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

    cur_dir = os.path.dirname(__file__)
    doc_stream = stream_docs(path=os.path.join(cur_dir, 'movie_data.csv'))

#   pbar = pyprind.ProgBar(45)

    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
#        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    pickle.dump(clf, open(clf_path, 'wb'), protocol=4)            