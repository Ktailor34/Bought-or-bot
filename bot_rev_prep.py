from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd


analyze = SentimentIntensityAnalyzer()


dataset = pd.read_csv("Bot reviews.csv", header = None)
X = dataset.iloc[:, 4:]
reviews = X.astype(str).values.tolist()


def sentiment_analyzer_scores(review):
	score = analyze.polarity_scores(review)
	return score

Sentiment=[]
for i in reviews:
    score= sentiment_analyzer_scores(i)
    compound = score['compound']
    #positive
    if(compound >= 0.05):
        Sentiment.append('pos')
        
    #neutral
    if((compound > -0.05) and (compound < 0.05)):
        Sentiment.append('neu')
        
    #negative
    if(compound <= -0.05):
        Sentiment.append('neg')

dataset['reviewType'] = Sentiment

dataset.to_csv('bot_reviews.csv')







    
   
    




