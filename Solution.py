from itertools import groupby
#22 23 17 18 19 20 21
import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')
nltk.download('vader_lexicon')
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
# % matplotlib inline

df = pd.read_csv('demonetization-tweets.csv', encoding='ISO-8859-1')
df['created'] = pd.to_datetime(df['created'])
df['Date'] = df['created'].dt.strftime('%m/%d/%Y')
df['month']=df['created'].map(lambda x: x.strftime('%m'))
df['day']=df['created'].map(lambda x: x.strftime('%d'))

df_filtered=df[df['month']=='04']
#21,22 11-22
df_filtered=df_filtered[df_filtered['day']=='18']
dm_df=df_filtered





demonitization_df = dm_df[['text']]
print(demonitization_df.head())


def clean_tweets(tweet):
    # Remove Html
    tweet = BeautifulSoup(tweet).get_text()

    pattern = re.compile(r"(RT) @[^\s]+[\s]?|@[^\s]+[\s]?")

    tweet = re.sub(
        pattern,
        "", tweet)

    # Remove Non-Letters
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)

    # Convert to lower_case and split
    tweet = tweet.lower().split()

    # Remove stopwords
    stop = set(stopwords.words('english'))
    stop.add('https')
    words = [w for w in tweet if not w in stop]

    # join the words back into one string
    return (' '.join(words))


demonitization_df['processedtext'] = demonitization_df['text'].apply(lambda x: clean_tweets(x))
demonitization_df.head()
sid = SentimentIntensityAnalyzer()
demonitization_df['sentiment_compound_polarity'] = demonitization_df.processedtext.apply(
    lambda x: sid.polarity_scores(x)['compound'])
demonitization_df['sentiment_neutral'] = demonitization_df.processedtext.apply(lambda x: sid.polarity_scores(x)['neu'])
demonitization_df['sentiment_negative'] = demonitization_df.processedtext.apply(lambda x: sid.polarity_scores(x)['neg'])
demonitization_df['sentiment_pos'] = demonitization_df.processedtext.apply(lambda x: sid.polarity_scores(x)['pos'])
demonitization_df['sentiment_type'] = ''

#The Compound score is a metric that calculates the sum of all the lexicon
# ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive).

demonitization_df.loc[demonitization_df.sentiment_compound_polarity > 0, 'sentiment_type'] = 'POSITIVE'
demonitization_df.loc[demonitization_df.sentiment_compound_polarity == 0, 'sentiment_type'] = 'NEUTRAL'
demonitization_df.loc[demonitization_df.sentiment_compound_polarity < 0, 'sentiment_type'] = 'NEGATIVE'
demonitization_df.head()
demonitization_df.to_csv ('export_dataframe.csv', index = False, header=True) #Don't forget to add '.csv' at the end of the path
print(demonitization_df['sentiment_type'].value_counts())












#
# #Divide the data in 80:20 based on sentiment_type
# processed_dmtweet_train, processed_dmtweet_test, sentimentdm_train, sentimentdm_test = train_test_split(
#      demonitization_df['processedtext'], demonitization_df['sentiment_type'], test_size=0.2, random_state=101)

# print(len(processed_dmtweet_train), len(processed_dmtweet_test),
#      len(processed_dmtweet_train) + len(processed_dmtweet_test))
#
# ##vec = CountVectorizer(min_df=2  , ngram_range=(1,3)).fit(demonitization_df['processedtext'])
# #
# # # bag_of_words = vec.transform(demonitization_df['processedtext'])
# # # sum_words = bag_of_words.sum(axis=0)
# # # words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
# # # words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)[:10]
# # # print(words_freq)
#
# # #corpus = [
# # ...     'This is the first document.',
# # ...     'This document is the second document.',
# # ...     'And this is the third one.',
# # ...     'Is this the first document?',
# # ... ]
# #['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
# # [[0 1 1 1 0 0 1 0 1]
# #  [0 2 0 1 0 1 1 0 1]
# #  [1 0 0 1 1 0 1 1 1]
# #  [0 1 1 1 0 0 1 0 1]]
# count_vect = CountVectorizer(min_df=2, ngram_range=(1, 3))
# X_dm = count_vect.fit_transform(demonitization_df['processedtext'])
# X_dmTrain_counts = count_vect.fit_transform(processed_dmtweet_train)
# X_dmTest_counts = count_vect.transform(processed_dmtweet_test)
#
# #
# print('Train data-Shape of Sparse Matrix: ', X_dmTrain_counts.shape)
# print('Train data-Amount of Non-Zero occurences: ', X_dmTrain_counts.nnz)
# #
# print('Test data-Shape of Sparse Matrix: ', X_dmTest_counts.shape)
# print('Test data-Amount of Non-Zero occurences: ', X_dmTest_counts.nnz)
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# #. The lower the IDF value of a word, the less unique it is to any particular document. Basically weights of each word.Internally this is computing the tf * idf  multiplication where your term frequency is weighted by its IDF values
# tfidf_transformer = TfidfTransformer()
# X_dm_tfidf = tfidf_transformer.fit_transform(X_dm)
# X_dmTrain_tfidf = tfidf_transformer.fit_transform(X_dmTrain_counts)
# X_dmTest_tfidf = tfidf_transformer.transform(X_dmTest_counts)
# df = pd.DataFrame(X_dmTrain_tfidf.toarray(), columns = count_vect.get_feature_names())
# df.to_csv ('export_dataframe2.csv', index = False, header=True) #Don't forget to add '.csv' at the end of the path
#
#
#
# print(X_dmTrain_tfidf.shape)
# print(X_dmTest_tfidf.shape)
#
# demonetization_prediction = {}
#
# from sklearn.linear_model import LogisticRegression
#
# logistic_regression_model_dm = LogisticRegression(random_state=101)
# logistic_regression_model_dm.fit(X_dmTrain_tfidf, sentimentdm_train)
# demonetization_prediction['Logistic Regression'] = logistic_regression_model_dm.predict(X_dmTest_tfidf)
# print("Demonetization-Logistic regression Accuracy : {}".format(
#     logistic_regression_model_dm.score(X_dmTest_tfidf, sentimentdm_test)))
#
# keys = demonetization_prediction.keys()
# for key in keys:
#     print(" {}:".format(key))
#     print(metrics.classification_report(sentimentdm_test, demonetization_prediction.get(key),
#                                         target_names=["POSITIVE", "NEGATIVE", "NEUTRAL"]))
#     print("\n")
#
# logistic_regression_model_dm1 = LogisticRegression(random_state=101)
# logistic_regression_model_dm1.fit(X_dm_tfidf, demonitization_df['sentiment_type'])
# demonitization_df['Sentiment_byLR'] = logistic_regression_model_dm1.predict(X_dm_tfidf)
# demonitization_df.head()