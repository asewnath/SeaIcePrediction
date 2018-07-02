"""

Sea Ice Prediction Model Test Bench

Author:  Akira Sewnath
Purpose: Script for testing different supervised learning models
         
"""


import forecast_funcs as ff
import statsmodels.api as sm
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
weight = 0

rawDataPath = '../../Data/' 
derivedDataPath = '../../DataOutput/'

yrForecast = 2015

#thickness = np.ma.mean(ff.get_pmas_month(rawDataPath, yrForecast, forecastMonth))
thick, forecastThickMean = ff.get_ice_thickness(rawDataPath, startYr, yrForecast, forecastMonth)

#Arctic Data
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

'''
_, unweightedPredVarForecast, predVarTrain, predVarForecast = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                 varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                 region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weight)
'''

_, unweightedPredVarForecast, predVarTrainMean, predVarTrainMed, predVarForecastMean, predVarForecastMed = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                                                                 varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                                                                 region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weight)


thickMean = np.reshape(thick, (-1, 1))
predVarTrainMean = np.reshape(predVarTrainMean, (-1, 1))
predVarTrainMed = np.reshape(predVarTrainMed, (-1, 1))
sqrMean = np.power(predVarTrainMean, 2)
sqrMed = np.power(predVarTrainMed, 2)
multFeat = np.multiply(predVarTrainMean, predVarTrainMed)

predVarTrain = np.concatenate((predVarTrainMean, sqrMean, predVarTrainMed, sqrMed, multFeat, thickMean), axis=1)
#predVarTrain = np.concatenate((predVarTrainMean, predVarTrainMed, thickMean), axis=1)

predVarForecast = np.array([predVarForecastMean, predVarForecastMean**2, predVarForecastMed, predVarForecastMed**2, predVarForecastMean*predVarForecastMed, forecastThickMean])
#predVarForecast = np.array([predVarForecastMean, predVarForecastMed, forecastThickMean,])
predVarForecast = np.reshape(predVarForecast, (1, -1))


#do random forest regressor
#until I produce more features, any feature argument here is going to be useless
extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2]))
final_val = 0
score = 0

#while final_val < 4.6 or score < 0.60:
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(predVarTrain, extentDetrendTrain, test_size=0.1, random_state=None) #cross validation going on here
    
    regr = RandomForestRegressor(verbose=0, n_estimators=30, bootstrap=True, criterion="mse", max_features=2,
                                 max_depth=4, min_samples_split=8, min_samples_leaf=5,
                                 min_weight_fraction_leaf=0, max_leaf_nodes=4,
                                 min_impurity_decrease=0, oob_score=False, n_jobs=1,
                                 random_state=None)
    
    #regr.fit(predVarTrain, extentDetrendTrain)
    regr.fit(X_train, y_train)
    score = regr.score(X_test, y_test)
    print(score)
    randForestPred = regr.predict(predVarForecast)
    
    
    
    #ridge = Ridge(solver='auto')
    #ridge.fit(X_train, y_train)
    #ridgePred = ridge.predict(predVarForecast)
    
    
    mlp = MLPRegressor(hidden_layer_sizes=(300,200,200,200), max_iter=1000, alpha=0.00001, batch_size= 'auto',
                       early_stopping=False, activation='relu')
    mlp.fit(X_train, y_train)
    print(mlp.score(X_test, y_test))
    mlpPred = mlp.predict(predVarForecast)
    
    '''
    predForecastData = [1]
    predForecastData.append(predVarForecast)
    predTrainData = np.ones((size(yrsTrain)))
    predTrainData = np.column_stack((predTrainData, array(predVarTrain)))    
     
    
    model=sm.OLS(extentDetrendTrain, predTrainData)
    fit=model.fit()
    extentDetrendForecast = fit.predict(predForecastData)[0]
    '''
    
    #extentForrAbs = extentDetrendForecast + extentTrendPersist
    
    print ('Observed extent: ',extentYr[-1])
    #print ('Linear Regression Forecast extent: ',extentForrAbs)
    final_val = randForestPred[0] + extentTrendPersist
    print ('Random Forest Regression Forecast extent: ', final_val)
    #print ('Ridge Regression Prediction: ', ridgePred[0] + extentTrendPersist)
    print ('MLP Prediction: ', mlpPred[0] + extentTrendPersist)
    print("\n")
    
joblib.dump(regr, 'randForReg.pkl', protocol=2)
#joblib.dump(ridge, 'ridgeReg.pkl', protocol=2)
joblib.dump(mlp, 'mlpReg.pkl', protocol=2)



