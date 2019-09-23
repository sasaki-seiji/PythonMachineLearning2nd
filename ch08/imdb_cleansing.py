import re

from load_movie_data import df

df.loc[0, 'review'][-50:]
# 2019.09.22 add
if __name__ == '__main__':
    print(df.loc[0, 'review'][-50:])


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text




preprocessor(df.loc[0, 'review'][-50:])
# 2019.09.22 add
if __name__ == '__main__':
    print(preprocessor(df.loc[0, 'review'][-50:]))


preprocessor("</a>This :) is :( a test :-)!")
# 2019.09.22 add
if __name__ == '__main__':
    print(preprocessor("</a>This :) is :( a test :-)!"))

df['review'] = df['review'].apply(preprocessor)
# 2019.09.22 add
if __name__ == '__main__':
    print(preprocessor(df.loc[0, 'review']))
