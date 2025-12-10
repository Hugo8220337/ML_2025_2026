import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')



def preprocessing(df):
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)
    df = df.replace(to_replace=r'\d', value='', regex=True)
    df = df.map(lambda x: word_tokenize(x) if isinstance(x, str) else x)
    print('\n------------------------------------------------------------------------------\n')
    print(df['data'][0])
    stop_words = set(stopwords.words('english'))
    df = df.map(lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) else x)
    print('\n------------------------------------------------------------------------------\n')
    print(df['data'][0])
    print('\n------------------------------------------------------------------------------\n')

    wl = WordNetLemmatizer()
    df = df.map(lambda x: pos_tag(x) if isinstance(x, list) else x)
    df = df.map(lambda x: [wl.lemmatize(word, get_wordnet_pos(pos)) for word, pos in x] if isinstance(x, list) else x)

    
    return df

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)



