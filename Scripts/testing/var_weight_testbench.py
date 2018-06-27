"""

Sea Ice Prediction Variable Weighting Test Bench

Author:  Akira Sewnath
Purpose: Data collection for experimental design of variable weighting
         techniques
         
"""

import forecast_funcs as ff
import statsmodels.api as sm
import numpy as np
import xlsxwriter
from sklearn.externals import joblib
from pylab import size
from pylab import array

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

regr = joblib.load('randForReg.pkl')
ridge = joblib.load('ridgeReg.pkl')
mlp = joblib.load('mlpReg.pkl')

#Create workbook to record experiment data
workbook = xlsxwriter.Workbook('var_weight_experiment_data.xlsx')


#Increasing forecast increments (record weighted and unweighted for our information)
worksheet = workbook.add_worksheet()
worksheet.name = "Increasing Year Forecast"

row = 0
col = 0

worksheet.write(row, col,   'Forecast Year')
worksheet.write(row, col+1, 'Years Behind Forecast Year')
worksheet.write(row, col+2, 'Observed')
worksheet.write(row, col+3, 'Correlation Unweighted')
worksheet.write(row, col+4, 'Correlation Weighted')
worksheet.write(row, col+5, 'Random Forest')
worksheet.write(row, col+6, 'Ridge Regression')
worksheet.write(row, col+7, 'MLP Regression')

row += 1

for index in range(yrsAvailable - minYrAdd + 1) :
    
    yrForecast = startYr + minYrAdd + index
    yrsTrain, extentTrain = ff.get_ice_extentN(rawDataPath, predMonth, startYr, 
                            yrForecast-1, icetype=iceType, version=siiVersion, 
                            hemStr=hemStr)
    extentDetrendTrain, lineTrain = ff.get_varDT(yrsTrain, extentTrain)
    
    varTrain = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth, 
                              yrsTrain, hemStr)
    varForecast = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth,
                                 array(yrForecast), hemStr)

    worksheet.write(row, col, yrForecast)
    worksheet.write(row, col+1, yrForecast-startYr)
    
    years, extentYr = ff.get_ice_extentN(rawDataPath, predMonth, yrForecast, yrForecast, icetype=iceType, version=siiVersion, hemStr=hemStr)
    observed = extentYr[-1]
    worksheet.write(row, col+2, observed)

    for weightBool in range(2):
        
        _, unweightedPredVarTrain, predVarTrain, predVarForecast = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                         varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                         region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weightBool)
        predForecastData = [1]
        predForecastData.append(predVarForecast)
        predTrainData = np.ones((size(yrsTrain)))
        predTrainData = np.column_stack((predTrainData, array(predVarTrain)))    
          
        model=sm.OLS(extentDetrendTrain, predTrainData)
        fit=model.fit()
        extentDetrendForecast = fit.predict(predForecastData)[0]
    
        extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2]))
        extentForrAbs = extentDetrendForecast + extentTrendPersist
        
        if(weightBool == 0):
            worksheet.write(row, col+3, extentForrAbs)
        else:
            worksheet.write(row, col+4, extentForrAbs)
    
    predVarForecast = np.array([predVarForecast, predVarForecast**2])
    predVarForecast = np.reshape(predVarForecast, (1, -1))
    
    randForestPred = regr.predict(predVarForecast)[0]
    worksheet.write(row, col+5, randForestPred + extentTrendPersist)
    
    
    #ridgePred = ridge.predict(predVarForecast)[0]
    #worksheet.write(row, col+6, ridgePred + extentTrendPersist)
    
    mlpPred = mlp.predict(predVarForecast)[0]
    worksheet.write(row, col+7, mlpPred + extentTrendPersist)
   
    print (index)
    row += 1
    
'''
#6 year increments (sliding window)
worksheet = workbook.add_worksheet()
worksheet.name = "6 Year Sliding Window Forecast"

row = 0
col = 0

worksheet.write(row, col,   'Forecast Year')
worksheet.write(row, col+1, 'Years Behind Forecast Year')
worksheet.write(row, col+2, 'Observed')
worksheet.write(row, col+3, 'Correlation Unweighted')
worksheet.write(row, col+4, 'Correlation Weighted')

row += 1

window = 6
yrForecast = 1980 + window - 1

for index in range(yrsAvailable - window + 1) :
    
    yrForecast += 1
    startYr = yrForecast - window
    
    yrsTrain, extentTrain = ff.get_ice_extentN(rawDataPath, predMonth, startYr, 
                            yrForecast-1, icetype=iceType, version=siiVersion, 
                            hemStr=hemStr)
    extentDetrendTrain, lineTrain = ff.get_varDT(yrsTrain, extentTrain)
    
    varTrain = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth, 
                              yrsTrain, hemStr)
    varForecast = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth,
                                 array(yrForecast), hemStr)

    worksheet.write(row, col, yrForecast)
    worksheet.write(row, col+1, yrForecast-startYr)
    
    years, extentYr = ff.get_ice_extentN(rawDataPath, predMonth, yrForecast, yrForecast, icetype=iceType, version=siiVersion, hemStr=hemStr)
    observed = extentYr[-1]
    worksheet.write(row, col+2, observed)

    for weightBool in range(2):
        
        _, unweightedPredVarTrain, predVarTrain, predVarForecast = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                         varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                         region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weightBool)
        predForecastData = [1]
        predForecastData.append(predVarForecast)
        predTrainData = np.ones((size(yrsTrain)))
        predTrainData = np.column_stack((predTrainData, array(predVarTrain)))    
          
        model=sm.OLS(extentDetrendTrain, predTrainData)
        fit=model.fit()
        extentDetrendForecast = fit.predict(predForecastData)[0]
    
        extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2]))
        extentForrAbs = extentDetrendForecast + extentTrendPersist
        
        if(weightBool == 0):
            worksheet.write(row, col+3, extentForrAbs)
        else:
            worksheet.write(row, col+4, extentForrAbs)
    
    row += 1
    print (index)


#10 year increments (sliding window)
worksheet = workbook.add_worksheet()
worksheet.name = "10 Year Sliding Window Forecast"

row = 0
col = 0

worksheet.write(row, col,   'Forecast Year')
worksheet.write(row, col+1, 'Years Behind Forecast Year')
worksheet.write(row, col+2, 'Observed')
worksheet.write(row, col+3, 'Correlation Unweighted')
worksheet.write(row, col+4, 'Correlation Weighted')

row += 1

window = 10
yrForecast = 1980 + window - 1


for index in range(yrsAvailable - window + 1) :
    
    yrForecast += 1
    startYr = yrForecast - window
    
    yrsTrain, extentTrain = ff.get_ice_extentN(rawDataPath, predMonth, startYr, 
                            yrForecast-1, icetype=iceType, version=siiVersion, 
                            hemStr=hemStr)
    extentDetrendTrain, lineTrain = ff.get_varDT(yrsTrain, extentTrain)
    
    varTrain = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth, 
                              yrsTrain, hemStr)
    varForecast = ff.get_gridvar(derivedDataPath, forecastVar, forecastMonth,
                                 array(yrForecast), hemStr)

    worksheet.write(row, col, yrForecast)
    worksheet.write(row, col+1, yrForecast-startYr)
    
    years, extentYr = ff.get_ice_extentN(rawDataPath, predMonth, yrForecast, yrForecast, icetype=iceType, version=siiVersion, hemStr=hemStr)
    observed = extentYr[-1]
    worksheet.write(row, col+2, observed)

    for weightBool in range(2):
        
        _, unweightedPredVarTrain, predVarTrain, predVarForecast = ff.GetWeightedPredVar(derivedDataPath, yrsTrain, yrForecast, extentDetrendTrain, 
                                                                                         varTrain, varForecast, forecastVar, forecastMonth, predMonth, startYr, numYrsRequired, 
                                                                                         region, hemStr, iceType, normalize=0, outWeights=outWeights, weight=weightBool)
        predForecastData = [1]
        predForecastData.append(predVarForecast)
        predTrainData = np.ones((size(yrsTrain)))
        predTrainData = np.column_stack((predTrainData, array(predVarTrain)))    
          
        model=sm.OLS(extentDetrendTrain, predTrainData)
        fit=model.fit()
        extentDetrendForecast = fit.predict(predForecastData)[0]
    
        extentTrendPersist = (lineTrain[-1]+(lineTrain[-1]-lineTrain[-2]))
        extentForrAbs = extentDetrendForecast + extentTrendPersist
        
        if(weightBool == 0):
            worksheet.write(row, col+3, extentForrAbs)
        else:
            worksheet.write(row, col+4, extentForrAbs)
    
    row += 1
    print (index)
'''    
    

workbook.close()  






