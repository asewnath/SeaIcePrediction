# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:40:07 2018

@author: Akira
"""
#FIX THE CORRELATION NAMES IF NEEDED BECAUSE IT'S CONFUSING
#Detrended vs. Raw Ground Truth experimentation

from numpy import linalg as LA
import forecast_funcs as ff
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from pylab import size
from pylab import array

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

startYr = 1980
numYrsRequired = 5
minYrAdd = 6 
yrsAvailable = 36
iceType = 'area'
siiVersion ='v3.0'
forecastVar = 'conc'
forecastMonth = 6
outWeights = 1
predMonth = 9
region = 0
hemStr = 'N'
anomObs = 1
weight = 1

rawDataPath = '../../Data/' 
derivedDataPath = '../../DataOutput/'

yrForecast = 2015
randSeedNum = 50

thick, forecastThickMean = ff.get_ice_thickness(rawDataPath, startYr, yrForecast, forecastMonth)

yrsTrain, extentTrain = ff.get_ice_extentN(rawDataPath, predMonth, startYr, 
                        yrForecast-1, icetype=iceType, version=siiVersion, 
                        hemStr=hemStr)

extentDetrendTrain, lineTrain = ff.get_varDT(yrsTrain, extentTrain)

varTrain = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth, 
                          yrsTrain, hemStr)
varForecast = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth,
                             array(yrForecast), hemStr)
years, extentYr = ff.get_ice_extentN(rawDataPath, predMonth, yrForecast, yrForecast, icetype=iceType, version=siiVersion, hemStr=hemStr)
observed = extentYr[-1]
_, unweightedPredVarForecast, predVarTrainMean, predVarTrainMed, predVarForecastMean, predVarForecastMed = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                                                                 varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                                                                 region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weight)

extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2])) #add to detrended forecast predictions

thickMean = np.reshape(thick, (-1, 1))
predVarTrainMean = np.reshape(predVarTrainMean, (-1, 1))

predVarTrain = predVarTrainMean
#predVarTrain = np.concatenate((predVarTrainMean, thickMean), axis=1)

predVarForecast = predVarForecastMean
#predVarForecast = np.array([predVarForecastMean, forecastThickMean])
#predVarForecast = np.reshape(predVarForecast, (1, -1))


detrendMLPCorr = []
rawMLPCorr = []
detrendRandForrCorr = []
rawRandForrCorr = []

detrendMLPPred = []
rawMLPPred = []
detrendRandForrPred = []
rawRandForrPred = []

rawMLPCoDFilt = []

randomState=2

#Working with raw data
X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentTrain, test_size=0.3, random_state=randomState)

for randomSeed in range(randSeedNum):
        '''    
        regr = RandomForestRegressor(verbose=0, n_estimators=30, bootstrap=True, criterion="mse", max_features=2,
                                 max_depth=4, min_samples_split=8, min_samples_leaf=5,
                                 min_weight_fraction_leaf=0, max_leaf_nodes=4,
                                 min_impurity_decrease=0, oob_score=False, n_jobs=1,
                                 random_state=randomSeed)
        '''
        regr = RandomForestRegressor(random_state=randomSeed)
        
        regr.fit(X_train, y_train)
        rawRandForrCorr.append(regr.score(X_test, y_test))
        rawRandForrPred.append(regr.predict(predVarForecast)[0])
        
        '''
        mlp = MLPRegressor(hidden_layer_sizes=(6,3), max_iter=800, alpha=0.001, batch_size= 'auto',
                       early_stopping=False, activation='relu', random_state=randomSeed)
        '''
        mlp = MLPRegressor(random_state=randomSeed,hidden_layer_sizes=(3,))
        #mlp = MLPRegressor(random_state=randomSeed)
        
        mlp.fit(X_train, y_train)
        score = mlp.score(X_test, y_test)
        rawMLPCorr.append(score)
        
        if(score > -0.1):
            rawMLPCoDFilt.append(score)
        
        rawMLPPred.append(mlp.predict(predVarForecast)[0])
        print(randomSeed)


#Working with detrended data
X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentDetrendTrain, test_size=0.3, random_state=randomState)

for randomSeed in range(randSeedNum):
        
        '''
        regr = RandomForestRegressor(verbose=0, n_estimators=30, bootstrap=True, criterion="mse", max_features=2,
                                 max_depth=4, min_samples_split=8, min_samples_leaf=5,
                                 min_weight_fraction_leaf=0, max_leaf_nodes=4,
                                 min_impurity_decrease=0, oob_score=False, n_jobs=1,
                                 random_state=randomSeed)
        '''
        regr = RandomForestRegressor(random_state=randomSeed)
        
        regr.fit(X_train, y_train)
        detrendRandForrCorr.append(regr.score(X_test, y_test))
        detrendRandForrPred.append(regr.predict(predVarForecast)[0] + extentTrendPersist)
        
        '''
        mlp = MLPRegressor(hidden_layer_sizes=(6,3), max_iter=800, alpha=0.001, batch_size= 'auto',
                       early_stopping=False, activation='relu', random_state=randomSeed)
        '''
        mlp = MLPRegressor(random_state=randomSeed, hidden_layer_sizes=(3,))
        #mlp = MLPRegressor(random_state=randomSeed)
        
        mlp.fit(X_train, y_train)
        detrendMLPCorr.append(mlp.score(X_test, y_test))
        detrendMLPPred.append(mlp.predict(predVarForecast)[0] + extentTrendPersist)
        print(randomSeed)

'''
pltRange = np.arange(0, np.size(extentTrain))
plt.scatter(pltRange, extentTrain)
plt.ylabel('raw ground truth')
plt.xlabel('Range')
plt.title('MLP Raw CoD Filtered Scores')
plt.savefig('mlp_raw_filtered.png')
plt.close()
'''
#Create filtered CoD scatter plot for MLP
'''        
pltRange = np.arange(0, np.size(rawMLPCoDFilt))
plt.scatter(pltRange, rawMLPCoDFilt)
plt.ylabel('CoD Scores')
plt.xlabel('Range')
plt.title('MLP Raw CoD Filtered Scores')
plt.savefig('mlp_raw_filtered.png')
plt.close()
'''

#Creating scatter plots of the data
randSeedRange = np.arange(0, 50)
plt.scatter(randSeedRange, detrendMLPCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('MLP Detrended CoD Scores vs. Random Seeds')
plt.savefig('mlp_detrended_corr.png')
plt.close()

plt.scatter(randSeedRange, rawRandForrCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('Random Forest Raw CoD Scores vs. Random Seeds')
plt.savefig('rand_forr_raw_corr.png')
plt.close()

plt.scatter(randSeedRange, rawMLPCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('MLP Raw CoD Scores vs. Random Seeds')
plt.savefig('mlp_raw_corr.png')
plt.close()

plt.scatter(randSeedRange, detrendRandForrCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('Random Forest Detrended CoD Scores vs. Random Seeds')
plt.savefig('rand_forr_detrended_corr.png')
plt.close()


plt.hlines(observed, 0, 50, label='3.42')
plt.legend()
plt.scatter(randSeedRange, detrendMLPPred, c='m')
plt.ylabel('MLP Predictions')
plt.xlabel('Random Seeds')
plt.title('MLP Detrended Predictions for 2015 vs. Random Seeds')
plt.savefig('mlp_detrended_pred.png')
plt.close()

plt.hlines(observed, 0, 50, label='3.42')
plt.legend()
plt.scatter(randSeedRange, detrendRandForrPred, c='m')
plt.ylabel('Random Forest Predictions')
plt.xlabel('Random Seeds')
plt.title('Random Forest Detrended Predictions for 2015 vs. Random Seeds')
plt.savefig('rand_for_detrended_pred.png')
plt.close()

plt.hlines(observed, 0, 50, label='3.42')
plt.legend()
plt.scatter(randSeedRange, rawRandForrPred, c='m')
plt.ylabel('Random Forest Predictions')
plt.xlabel('Random Seeds')
plt.title('Random Forest Raw Predictions for 2015 vs. Random Seeds')
plt.savefig('rand_for_raw_pred.png')
plt.close()

plt.hlines(observed, 0, 50, label='3.42')
plt.legend()
plt.scatter(randSeedRange, rawMLPPred, c='m')
plt.ylabel('MLP Prediction')
plt.xlabel('Random Seeds')
plt.title('MLP Raw Predictions for 2015 vs. Random Seeds')
plt.savefig('mlp_raw_pred.png')
plt.close()









