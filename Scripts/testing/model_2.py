#sequential model testing: model 2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from model_2_data_collection import data_collection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#data preprocessing
feat, gTruth = data_collection()
gTruth=np.reshape(gTruth, (np.size(gTruth),))

#do some PCA
#pca = PCA(n_components=8, whiten=True )
#feat = pca.fit_transform(feat)
#joblib.dump(pca, 'seq_pca_2.pkl', protocol=2)
#xRange = np.arange(10, 200, 10)
#xRange = np.arange(1,15)
xRange = np.arange(25)


#model tuning
scores = []
scaler = StandardScaler()


for randomSeed in range(25):
    
    X_train, X_test, y_train, y_test = train_test_split(feat, 
                                                gTruth, test_size=0.2, 
                                                random_state=None)
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #for n in range(1, 15):
    '''
    mlp = RandomForestRegressor(random_state=randomSeed, n_estimators=30, max_features=10, 
                                         max_depth=10, min_samples_split=5, criterion='mae')
    ''' 
          
    mlp = MLPRegressor(random_state=None, hidden_layer_sizes=(100,50), 
                       max_iter=2000, activation='tanh', verbose = False,
                       solver='adam', early_stopping=True, alpha=0.0001)
    print(randomSeed)
    mlp.fit(X_train, y_train)
    
    if(mlp.score(X_test, y_test) >= 0.80):
        joblib.dump(mlp, 'seq_model_2.pkl', protocol=2)
        joblib.dump(scaler, 'scaler_extent.pkl', protocol=2)
    
    scores.append(mlp.score(X_test, y_test))  

#Creating scatter plots of the data
plt.scatter(xRange, scores, c='m')   
#plt.ylabel('CoD Scores')
#plt.xlabel('Neurons in first hidden layer')
#plt.xlabel('Runs')
#plt.title('MLP, solver=lbfgs, hidden_layer_size = (8,)')
#plt.savefig('mlp_sol.png')   
#plt.close()    
    