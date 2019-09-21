from nltk.corpus import stopwords

from tokenizer import tokenizer_porter

stop = stopwords.words('english')

[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]
#2019.09.22 add
if __name__ == '__main__':
    words = [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] 
    if w not in stop]
    print(words)
