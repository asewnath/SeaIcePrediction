
"""
Weighted and Unweighted Model Tuning

This script tunes an MLP and a Random Forest using weighted and unweighted data.
See the forecast functions file for more information for the weighting technique 
and generally what the functions do
"""



import forecast_funcs as ff
import matplotlib.pyplot as plt
import numpy as np
from pylab import array

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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
years, extentYr = ff.get_ice_extentN(rawDataPath, predMonth, yrForecast, 
                                     yrForecast, icetype=iceType, version=siiVersion, hemStr=hemStr)
observed = extentYr[-1]
extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2])) #add to detrended forecast predictions

regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')



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

#Features for testing data
regFeaturesTest = np.zeros((21, 1))    
for regInd in range(21):
    regionData = varForecast.data[yearInd]
    regionData[varForecast.mask[yearInd] == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    regFeaturesTest[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))        
            

predVarTrain = np.reshape(regFeaturesTrain, (35, 21))
predVarForecast = np.reshape(regFeaturesTest, (1, 21))

#CoD is the coefficient of determination that you retrieve when using the predict
#method in a sci-kit learn regression model. The documentation of these models
#will be very helpful

pltRawMLPCod  = []
pltRawMLPPred = []
for CVrandomState in range(5):
    
    rawMLPCoD  = []
    rawMLPPred = []
    X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentTrain, test_size=0.3, random_state=CVrandomState) #raw data

    for randomSeed in range(randSeedNum):
            
            mlp = MLPRegressor(random_state=randomSeed, hidden_layer_sizes=(20,), max_iter=400, 
                               activation='relu', alpha=0.001)
            
            mlp.fit(X_train, y_train)
            score = mlp.score(X_test, y_test)
            rawMLPCoD.append(score)
            rawMLPPred.append(mlp.predict(predVarForecast)[0])
            print(randomSeed)
            
    pltRawMLPCod.append(rawMLPCoD)
    pltRawMLPPred.append(rawMLPPred)        


#Create subplot for results
randSeedRange = np.arange(0, 50)
fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False)
ax = ax.flatten()

for ind in range(10):
    
    if(ind < 5):
        ax[ind].scatter(randSeedRange, pltRawMLPCod[ind])
        ax[ind].set_xlabel('Random Seeds')
        ax[ind].set_ylabel('CoD Scores')
        ax[ind].set_title('MLP Test CV' + str(ind))
    else:
        ax[ind].scatter(randSeedRange, pltRawMLPPred[ind-5], c='m')
        ax[ind].hlines(observed, 0, 50, label='4.62')
        ax[ind].legend()
        ax[ind].set_xlabel('Random Seeds')
        ax[ind].set_ylabel('MLP 2015 Predictions')
        ax[ind].set_title('MLP Test CV' + str(ind-5))





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

#Features for testing data
regFeaturesTest = np.zeros((21, 1))
for regInd in range(21):
    regionData = varForecast.data[yearInd]
    regionData[varForecast.mask[yearInd] == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    regFeaturesTest[regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))        
            
predVarTrain = np.reshape(regFeaturesTrain, (35, 21))
predVarForecast = np.reshape(regFeaturesTest, (1, 21))

pltRandForCod  = []
pltRandForPred = []
for CVrandomState in range(5):

    detrendRandForCod = []
    detrendRandForPred = []
    X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentDetrendTrain, test_size=0.3, random_state=CVrandomState)
    
    for randomSeed in range(randSeedNum):
            
            regr = RandomForestRegressor(random_state=randomSeed, n_estimators=20, max_features=15, 
                                         max_depth=5, min_samples_split=2, criterion='mse')
            regr.fit(X_train, y_train)
            detrendRandForCod.append(regr.score(X_test, y_test))
            detrendRandForPred.append(regr.predict(predVarForecast)[0] + extentTrendPersist)
            print(randomSeed)

    pltRandForCod.append(detrendRandForCod)
    pltRandForPred.append(detrendRandForPred)  

#Create subplot for results
randSeedRange = np.arange(0, 50)
fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False)
ax = ax.flatten()

for ind in range(10):
    
    if(ind < 5):
        ax[ind].scatter(randSeedRange, pltRandForCod[ind])
        ax[ind].set_xlabel('Random Seeds')
        ax[ind].set_ylabel('CoD Scores')
        ax[ind].set_title('RF Test CV' + str(ind))
    else:
        ax[ind].scatter(randSeedRange, pltRandForPred[ind-5], c='m')
        ax[ind].hlines(observed, 0, 50, label='4.62')
        ax[ind].legend()
        ax[ind].set_xlabel('Random Seeds')
        ax[ind].set_ylabel('RF 2015 Predictions')
        ax[ind].set_title('RF Test CV' + str(ind-5))







