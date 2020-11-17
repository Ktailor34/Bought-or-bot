from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd


analyze = SentimentIntensityAnalyzer()


dataset = pd.read_csv("Bot reviews.csv", header = None)
X = dataset.iloc[:, 4:]
reviews = X.astype(str).values.tolist()


def sentiment_analyzer_scores(review):
	score = analyze.polarity_scores(review)
	return str(score) 

A = []
for i in reviews:
	A.append(sentiment_analyzer_scores(i))

dataset["score"] = A



#sentiment_analyzer_scores(j)
	
