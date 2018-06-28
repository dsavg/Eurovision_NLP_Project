import pandas as pd
import pycountry
from itertools import chain
import tweepy
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.grid_objs import Grid, Column
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)
from sklearn.feature_extraction import stop_words
import plotly.graph_objs as go
from collections import Counter
import re
import string
import nltk
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim import corpora, models
import gensim
from countryinfo import countries


def exclude_urls(text):
    """
    This function uses regex to get rid of any url pages
    the tweet may have
    """
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                  '', text, flags=re.MULTILINE)


def tokenize(text):
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3.
    """
    my_additional_stop_words = ['https', 'http', 'twitter', 'com',
                                'eurovision', 'eurovis', 'pic', 'th', 's']
    my_stop_words = stop_words.ENGLISH_STOP_WORDS.union(
        my_additional_stop_words)
    text = text.lower()
    text = exclude_urls(text)
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    token = nltk.word_tokenize(text)
    token = [w for w in token if len(w) > 1]
    token = [w for w in token if w not in my_stop_words]
    return token


def stemwords(words):
    """
    Given a list of tokens/words, returns a new list with each word
    stemmed using a PorterStemmer.
    """
    stemmer = PorterStemmer()
    words = [w.decode('ascii', 'ignore') for w in words]
    stemmed = [stemmer.stem(w) for w in words]
    fwords = [w.encode('ascii', 'ignore') for w in stemmed]
    return fwords


def tokenizer(text):
    """
    Tokenizes and Steams text
    """
    return ' '.join(stemwords(tokenize(text)))


def tweet_sentiment(df, column_name):
    """
    Performing sentiment analysis given a dataframe and a column name
    """
    list_unique_words = df[column_name].unique()
    analyzer = SentimentIntensityAnalyzer()

    dict_words = {}
    for word in list_unique_words:
        dict_words[word] = analyzer.polarity_scores(word)

    df['comp'] = df[column_name].map(dict_words)
    return pd.concat([df.drop(['comp'], axis=1),
                      df['comp'].apply(pd.Series)], axis=1)


def tweet_tfidf(df, column_name, num_terms):
    """
    Perfoming tfidf on dataframe given a dataframe,
    a column name and the number of term to be
    considered
    """
    tvec = TfidfVectorizer(tokenizer=tokenize,
                           min_df=.0025,
                           max_df=.1,
                           stop_words='english',
                           ngram_range=(1, 4))
    tvec_weights = tvec.fit_transform(df[column_name].dropna())
    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(),
                               'weight': weights})

    return weights_df.sort_values(by='weight', ascending=False).head(num_terms)


def country_feature(text):
    """
    Creates a country feature
    """
    country_list = []
    for i in countries:
        country_list.append(tokenizer(i['name']))
    countries_mentioned = []
    for i in text:
        if i in country_list:
            countries_mentioned.append(i)

    return countries_mentioned


def tweet_topic_modelling(df, column_name, topics):
    """
    Perfoming topic modelling on dataframe given
    a dataframe and a column name
    """
    dictionary = corpora.Dictionary(df[column_name].tolist())
    corpus = [dictionary.doc2bow(text) for text in df[column_name].tolist()]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics,
                                               id2word=dictionary,
                                               passes=20)
    topics = []
    for i, topic in enumerate(ldamodel.print_topics(num_topics=topics,
                                                    num_words=4)):
        words = topic[1].split("+")
        topics.append(' '.join([re.findall("[a-zA-Z]+", w)[0] for w in words]))
    return topics, ldamodel, corpus, dictionary
