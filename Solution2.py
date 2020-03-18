

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

import tweepy
import csv
import pandas as pd
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('demonetization-tweets.csv', encoding='ISO-8859-1')
df['created'] = pd.to_datetime(df['created'])
df['Date'] = df['created'].dt.strftime('%m/%d/%Y')

cdata=df[['Date','text']]

df1 = cdata.groupby('Date')['text'].apply(lambda x: ', '.join(x.astype(str))).reset_index()


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


df1['Tweets'] = df1['text'].apply(lambda x: clean_tweets(x))
cdata=df1


read_stock_p=pd.read_csv('INR_prices.csv')
cdata['Prices']=""
indx=0

for i in range (0,len(cdata)):
    for j in range (0,len(read_stock_p)):
        get_tweet_date=cdata.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]
        if(str(get_stock_date)==str(get_tweet_date)):
            indx=indx+1
            #print(get_stock_date," ",get_tweet_date)
            cdata.at[i,'Prices']=(read_stock_p.price.iloc[j])
            break

cdata["Comp"] = ''
cdata["Negative"] = ''
cdata["Neutral"] = ''
cdata["Positive"] = ''
import unicodedata
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in cdata.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', cdata.loc[indexx, 'Tweets'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        cdata.at[indexx, 'Comp']= sentence_sentiment['compound']
        cdata.at[indexx, 'Negative']= sentence_sentiment['neg']
        cdata.at[indexx, 'Neutral']= sentence_sentiment['neu']
        cdata.at[indexx, 'Positive']= sentence_sentiment['pos']
    except TypeError:
        print (cdata.loc[indexx, 'Tweets'])
        print (indexx)


posi=0
nega=0
for i in range (0,len(cdata)):
    get_val=cdata.Comp[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
posper=(posi/(len(cdata)))*100
negper=(nega/(len(cdata)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
#mlpt.pie(arr,labels=['positive','negative'])
# mlpt.plot()
# mlpt.show()
df_=cdata[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()

X_train, X_test, y_train, y_test = train_test_split(cdata[['Date','Comp','Negative','Neutral','Positive']], cdata[['Prices']], test_size=0.30, random_state=42)






sentiment_score_list = []
for date, row in X_train.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix

reg = LinearRegression()
reg.fit(numpy_df_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
sentiment_score_list = []
for date, row in X_test.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)
y_pred=reg.predict(numpy_df_test)

result = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print(result)

plt.plot(y_test.values.flatten(),'r')
plt.plot(y_pred.flatten(),'g')
plt.show()