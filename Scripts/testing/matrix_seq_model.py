#sequential model testing (matrix input)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
#from data_collection import data_collection
from data_collection import mat_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#data preprocessing
#feat, gTruth = data_collection()
feat, gTruth = mat_preprocess()

#do some PCA
#pca = PCA(n_components=14, whiten=False )
#feat = pca.fit_transform(feat)
#joblib.dump(pca, 'seq_pca_1.pkl', protocol=2)

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
    mlp = MLPRegressor(random_state=None, hidden_layer_sizes=(100,100,50,30), 
                       max_iter=500, activation='tanh', verbose = False,
                       solver='adam', alpha=0.0001, batch_size=1)
    mlp.fit(X_train, y_train)
    print(randomSeed)
    if(mlp.score(X_test, y_test) >= 0.85):
        joblib.dump(mlp, 'seq_model_1_mat.pkl', protocol=2)
        joblib.dump(scaler, 'conc_scale.pkl', protocol=2)
    
    scores.append(mlp.score(X_test, y_test))
  
       
#Creating scatter plots of the data
plt.scatter(xRange, scores, c='m')   
#plt.ylabel('CoD Scores')
#plt.xlabel('Neurons in first hidden layer')
#plt.xlabel('Runs')
#plt.title('MLP, solver=lbfgs, hidden_layer_size = (8,)')
#plt.savefig('mlp_sol.png')   
#plt.close()    
    