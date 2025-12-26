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


def _to_lower(df):
    return df.map(lambda x: x.lower() if isinstance(x, str) else x)

def _remove_punctuation(df):
    return df.replace(to_replace=r'[^\w\s]', value=' ', regex=True)

def _remove_digits(df):
    return df.replace(to_replace=r'\d', value='', regex=True)

def _tokenize(df):
    return df.map(lambda x: word_tokenize(x) if isinstance(x, str) else x)

def _remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    return df.map(lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) else x)

def _lemmatize(df):
    wl = WordNetLemmatizer()
    
    def lemmatize_list(token_list):
        if not isinstance(token_list, list): return token_list
        
        tagged = pos_tag(token_list)
        return [wl.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

    return df.map(lemmatize_list)

def preprocessing(
    df, 
    to_lower=True, 
    remove_punctuation=True, 
    remove_digits=True, 
    tokenize=True, 
    remove_stopwords=True, 
    lemmatize=True
):
    if to_lower:
        df = _to_lower(df)
    if remove_punctuation:
        df = _remove_punctuation(df)
    if remove_digits:
        df = _remove_digits(df)
    if tokenize:
        df = _tokenize(df)
    if remove_stopwords:
        df = _remove_stopwords(df)
    if lemmatize:
        df = _lemmatize(df)

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