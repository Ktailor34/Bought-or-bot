# Creating a DataFrame of given trainData dataset.
import pandas as pd

# Import train_test_split function
from sklearn.model_selection import train_test_split

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

trainData = pd.read_json('yelp_combined_dataset.json')
testData = pd.read_csv('bot_reviews.csv')

testData.drop(['text'], axis=1)

trainData = trainData.append(testData)


data=pd.DataFrame({
    # 'user_id':trainData['user_id'],
    'stars':trainData['stars'],
    # 'text':trainData['text'],
    'review_count':trainData['review_count'],
    # 'friends':trainData['friends'],
    'isFake':trainData['isFake'],
    'friendCount':trainData['friendCount'],
    'reviewType':trainData['reviewType']
})
data.head()

data['reviewType'] = [(-1 if value == 'neg' else (0 if value == 'neu' else 1)) for value in data['reviewType']]

print(data['isFake'])
f = open("test.txt", "a")
f.write((data['isFake'].to_string()))
f.close()


# x = data.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# data = pd.DataFrame(x_scaled)

# X=data[['user_id', 'stars', 'text','review_count','friends','friendCount','reviewType']]  # Features
X= data[['stars','review_count','friendCount','reviewType']]  # Features
y= data['isFake']  # Labels

trainData.feature_names=['stars','review_count','friendCount','reviewType']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=222) # 70% training and 30% test


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

print(X_train)
print(y_train)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Model Accuracy, how often is the classifier correct?
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


############## FIND BEST FEATURES
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

feature_imp = pd.Series(clf.feature_importances_,index=trainData.feature_names).sort_values(ascending=False)

print("important feats", feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()