#sequential model testing

import numpy as np
import matplotlib.pyplot as plt

from data_collection import data_collection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA


#data preprocessing
feat, gTruth = data_collection()

#do some PCA
pca = PCA(n_components=14, whiten=True )
feat = pca.fit_transform(feat)
#xRange = np.arange(10, 200, 10)
#xRange = np.arange(1,15)
xRange = np.arange(100)


#model tuning
scores = []

for randomSeed in range(100):
    
    X_train, X_test, y_train, y_test = train_test_split(feat, 
                                                gTruth, test_size=0.2, 
                                                random_state=randomSeed)
    #for n in range(1, 15):       
    mlp = MLPRegressor(random_state=randomSeed, hidden_layer_sizes=(8,), 
                       max_iter=200, activation='tanh', verbose = False,
                       solver='lbfgs', early_stopping=True, alpha=0.0001)
    mlp.fit(X_train, y_train)
    scores.append(mlp.score(X_test, y_test))
    
#Creating scatter plots of the data
plt.scatter(xRange, scores)
    
plt.ylabel('CoD Scores')
#plt.xlabel('Neurons in first hidden layer')
plt.xlabel('Random Seeds')
plt.title('MLP Random Seed=[0:100), solver=lbfgs, hidden_layer_size = (8,)')
#plt.savefig('mlp_sol.png')
    
#plt.close()    
    