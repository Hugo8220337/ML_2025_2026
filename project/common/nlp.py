import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


# Verify and download necessary NLTK resources (quietly)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)



def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocessing(df, remove_stopwords=True):
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df = df.replace(to_replace=r'[^\w\s]', value=' ', regex=True)
    df = df.replace(to_replace=r'\d', value='', regex=True)
    df = df.map(lambda x: word_tokenize(x) if isinstance(x, str) else x)
    
    # Remove stopwords if specified (by default, it's True)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        df = df.map(lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) else x)

    wl = WordNetLemmatizer()
    
    def lemmatize_list(token_list):
        if not isinstance(token_list, list): return token_list
        
        tagged = pos_tag(token_list)
        return [wl.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

    df = df.map(lemmatize_list)


    return df

def tfidf_vectorize(df, col_name='data', max_features=5000, lowercase=False, stop_words=None):
    """
    Perform TF-IDF vectorization on the specified column of the DataFrame.
    Serves as a feature extraction method for text data. 
    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        col_name (str): Name of the column to vectorize.
        max_features (int): Maximum number of features for TF-IDF.
        lowercase (bool): Whether to convert text to lowercase.
        stop_words (list or str): Stop words to remove during vectorization.
    Returns:
        tfidf_matrix (sparse matrix): TF-IDF feature matrix.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer instance.
    """
    corpus = df[col_name].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=lowercase,
        stop_words=stop_words
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix, vectorizer