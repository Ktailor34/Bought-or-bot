import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import pylab as pl


yelp_data = pd.read_json('yelp_combined_dataset.json')
bot_data = pd.read_csv('bot_reviews.csv')
yelp_data=yelp_data.loc[:500]
bot_data = bot_data.loc[:100]


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
f_array = [ 'stars', 'reviewType']
X= combined_data[f_array]  #features
Y= combined_data['isFake'].values #target

#train test split (with randomization)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .30, random_state= 5)


#create instances
knn_one = neighbors.KNeighborsClassifier(n_neighbors=1, p=2,  weights ='uniform',metric='euclidean').fit(x_train, y_train)
knn_two = neighbors.KNeighborsClassifier(n_neighbors=1, p=2,  weights ='distance',metric='euclidean').fit(x_train, y_train)
knn_three = neighbors.KNeighborsClassifier(n_neighbors=5, p=2,  weights ='uniform',metric='euclidean').fit(x_train, y_train)
knn_four = neighbors.KNeighborsClassifier(n_neighbors=5, p=2,  weights ='distance',metric='euclidean').fit(x_train, y_train)
knn_five = neighbors.KNeighborsClassifier(n_neighbors=15, p=2,  weights ='uniform',metric='euclidean').fit(x_train, y_train)
knn_six = neighbors.KNeighborsClassifier(n_neighbors=15, p=2,  weights ='distance',metric='euclidean').fit(x_train, y_train)
# # # # # # # # # # # # #
# PLOTTING CODE STARTS  #
# # # # # # # # # # # # #

# create a mesh to plot in
h=.02 # step size in the mesh
x_min, x_max = X[f_array[0]].min()-1, X[f_array[0]].max()+1
y_min, y_max = X[f_array[1]].min()-1, X[f_array[1]].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['KNN k=1 uniform',
          'KNN k=1 distance',
          'KNN k=5 uniform',
          'KNN k=5 distance',
          'KNN k=15 uniform',
          'KNN k=15 distance']

for i, clf in enumerate((knn_one, knn_two, knn_three, knn_four, knn_five, knn_six)):
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 3, i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    # Apply 
    pl.contourf(xx, yy, Z)
    pl.axis('tight')

    # Plot also the training points
    pl.scatter(X[f_array[0]], X[f_array[1]], c=Y, edgecolor='black')

    pl.title(titles[i])

pl.axis('tight')
pl.show()

