from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]




tokenizer('runners like running and thus they run')
# 2019.09.22 add
if __name__ == '__main__':
    print(tokenizer('runners like running and thus they run'))



tokenizer_porter('runners like running and thus they run')
# 2019.09.22 add
if __name__ == '__main__':
    print(tokenizer_porter('runners like running and thus they run'))
