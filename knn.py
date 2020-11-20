#implenting knn algorithm

#datastucture libraries
from sklearn.model_selection import train_test_split
import  numpy as np
import pandas as pd
import json

#modeling
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets

#analysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

yelp_data = pd.read_json('yelp_combined_dataset.json')
bot_data = pd.read_csv('bot_reviews.csv')

#drop not needed columns
yelp_data= yelp_data.drop(columns = ["user_id", "text", "friends"])
bot_data =bot_data.drop(columns = ["rowCount", "text"])

#combine data
combined_data = yelp_data.append(bot_data)


#change feature values    
change_to_int = {'N':-1, 'Y':1}
combined_data['isFake'] = combined_data['isFake'].replace(change_to_int)
change_to_int = {'pos':1, 'neu':0, 'neg':-1}
combined_data['reviewType'] = combined_data['reviewType'].replace(change_to_int)


#separate features
X= combined_data[['stars', 'review_count', 'reviewType', 'friendCount']]  #features
Y= combined_data[['isFake']] #target

#train test split (with randomization)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .30, random_state= 5)

#perform knn

clf = neighbors.KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
clf.fit(x_train, y_train.values.ravel())


#metric report
y_expect = y_test
y_pred = clf.predict(x_test)
print(classification_report(y_expect,y_pred))

#confusion matrix
print(confusion_matrix(y_expect, y_pred))






