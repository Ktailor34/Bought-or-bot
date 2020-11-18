#implenting knn algorithm

#datastucture libraries
import  numpy as np
import pandas as pd
import json

#modeling
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets

n_neighbors = 15

with open('yelp_combined_dataset.json') as f:
	data = json.load(f)

X = pd.DataFrame.from_dict(data)
Y = X['isFake']

X.drop(['isFake'], axis=1)

#X = features
#Y = target

neigh = KNeighborsClassifier(n_neighbors)
neigh.fit(X,Y)


