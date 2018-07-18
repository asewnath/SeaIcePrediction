# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:40:07 2018

@author: Akira
"""
#FIX THE CORRELATION NAMES IF NEEDED BECAUSE IT'S CONFUSING
#Region segmentation experimentation

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
iceType = 'extent'
siiVersion ='v3.0'
forecastVar = 'conc'
forecastMonth = 6
outWeights = 1
predMonth = 9
region = 0
hemStr = 'N'
anomObs = 1


rawDataPath = '../../Data/' 
derivedDataPath = '../../DataOutput/'

yrForecast = 2015
randSeedNum = 50

#thick, forecastThickMean = ff.get_ice_thickness(rawDataPath, startYr, yrForecast, forecastMonth)

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
extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2])) #add to detrended forecast predictions

regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')#ff.get_region_mask_sect(rawDataPath)


rawMLPCorr = []
rawMLPPred = []
detrendRandForrCorr = []
detrendRandForrPred = []

randomState=0



#Features for training data
regMean = np.zeros((21,1))
regFeaturesTrain = []

for yearInd in range(35):
    for regInd in range(21):
        regionData = varTrain.data[yearInd]
        regionData[varTrain.mask[yearInd] == True] = 0 #get rid of nan
        desiredRegion = regionMask == regInd
        regMean[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))
        
    regFeaturesTrain.append(regMean)

regFeaturesTest = np.zeros((21, 1))
#Features for testing data
for regInd in range(21):
    regionData = varForecast.data[yearInd]
    regionData[varForecast.mask[yearInd] == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    regFeaturesTest[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))        
            

predVarTrain = np.reshape(regFeaturesTrain, (35, 21))#regFeaturesTrain.reshape((35, 57, 57))
predVarForecast = np.reshape(regFeaturesTest, (1, 21))


#Working with raw data
X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentTrain, test_size=0.3, random_state=randomState)

for randomSeed in range(randSeedNum):
        
        mlp = MLPRegressor(random_state=randomSeed)
        
        mlp.fit(X_train, y_train)
        score = mlp.score(X_test, y_test)
        rawMLPCorr.append(score)
        rawMLPPred.append(mlp.predict(predVarForecast)[0])
        print(randomSeed)




#WEIGHTED

VarTrain, VarForecast = ff.get_weighted_var(yrsTrain, yrForecast, extentDetrendTrain, varTrain, varForecast, numYrsRequired)

#Features for training data
regMean = np.zeros((21,1))
regFeaturesTrain = []

for yearInd in range(35):
    for regInd in range(21):
        regionData = varTrain.data[yearInd]
        regionData[varTrain.mask[yearInd] == True] = 0 #get rid of nan
        desiredRegion = regionMask == regInd
        regMean[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))
        
    regFeaturesTrain.append(regMean)

regFeaturesTest = np.zeros((21, 1))
#Features for testing data
for regInd in range(21):
    regionData = varForecast.data[yearInd]
    regionData[varForecast.mask[yearInd] == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    regFeaturesTest[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))        
            

predVarTrain = np.reshape(regFeaturesTrain, (35, 21))#regFeaturesTrain.reshape((35, 57, 57))
predVarForecast = np.reshape(regFeaturesTest, (1, 21))



X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentDetrendTrain, test_size=0.3, random_state=randomState)

for randomSeed in range(randSeedNum):
        
        regr = RandomForestRegressor(random_state=randomSeed)
        regr.fit(X_train, y_train)
        detrendRandForrCorr.append(regr.score(X_test, y_test))
        detrendRandForrPred.append(regr.predict(predVarForecast)[0] + extentTrendPersist)





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
plt.scatter(randSeedRange, detrendRandForrCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('Random Forest Detrended CoD Scores vs. Random Seeds')
plt.savefig('rand_forr_detrended_corr.png')
plt.close()

plt.scatter(randSeedRange, rawMLPCorr)
plt.ylabel('CoD Scores')
plt.xlabel('Random Seeds')
plt.title('MLP Raw CoD Scores vs. Random Seeds')
plt.savefig('mlp_raw_corr.png')
plt.close()

plt.hlines(observed, 0, 50, label='4.62')
plt.legend()
plt.scatter(randSeedRange, detrendRandForrPred, c='m')
plt.ylabel('Random Forest Predictions')
plt.xlabel('Random Seeds')
plt.title('Random Forest Detrended Predictions for 2015 vs. Random Seeds')
plt.savefig('rand_for_detrended_pred.png')
plt.close()

plt.hlines(observed, 0, 50, label='4.62')
plt.legend()
plt.scatter(randSeedRange, rawMLPPred, c='m')
plt.ylabel('MLP Prediction')
plt.xlabel('Random Seeds')
plt.title('MLP Raw Predictions for 2015 vs. Random Seeds')
plt.savefig('mlp_raw_pred.png')
plt.close()









