import os
import nltk
import pickle
import pandas as pd
from scipy.sparse import hstack
from django.conf import settings
from nltk.corpus import stopwords
from .preprocessing import data_cleaning

STOPWORDS = set(stopwords.words('english'))

def clean_and_extract(data):

    df_test = pd.DataFrame(data = data, columns = ["text"])

    # Number of words
    df_test['num_words'] = df_test['text'].apply(lambda x: len(str(x).split()))

    # Number of capital_letters
    df_test['num_capital_let'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c.isupper()]))

    # Number of special characters
    df_test['num_special_char'] = df_test['text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()

    # Number of unique words
    df_test['num_unique_words'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

    # Number of numerics
    df_test['num_numerics'] = df_test['text'].apply(lambda x: sum(c.isdigit() for c in x))

    # Number of characters
    df_test['num_char'] = df_test['text'].apply(lambda x: len(str(x)))

    # Number of stopwords
    df_test['num_stopwords'] = df_test['text'].apply(lambda x: len([c for c in str(x).lower().split() if c in STOPWORDS]))

    df_test['preprocessed_question_text'] = df_test['text'].apply(lambda x : data_cleaning(x))

    tfidf = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/tfidf.sav", "rb"))
    X_test_ques = tfidf.transform(df_test['preprocessed_text'].values)

    count_vectorizer = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/count_vectorizer.sav", "rb"))
    X_test_ques_countV = count_vectorizer.transform(df_test['preprocessed_text'].values)
    
    num_words = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/num_words.sav", "rb"))
    num_unique_words = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/num_unique_words.sav", "rb"))
    num_char = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/num_char.sav", "rb"))
    num_stopwords = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/num_stopwords.sav", "rb"))

    X_test_num_words = num_words.fit_transform(df_test['num_words'].values.reshape(-1, 1))
    X_test_num_unique_words = num_unique_words.fit_transform(df_test['num_unique_words'].values.reshape(-1, 1))
    X_test_num_char = num_char.fit_transform(df_test['num_char'].values.reshape(-1, 1))
    X_test_num_stopwords = num_stopwords.fit_transform(df_test['num_stopwords'].values.reshape(-1, 1))

    X_te = hstack((
        X_test_ques,
        X_test_num_words,
        X_test_num_unique_words,
        X_test_num_char,
        X_test_num_stopwords
    ))

    X_te1 = hstack((
    X_test_ques_countV,
    X_test_num_words,
    X_test_num_unique_words,
    X_test_num_char,
    X_test_num_stopwords
    ))

    x = [X_te, X_te1, X_test_ques_countV]

    return x 
