#Script to test forecasting capabilities of ensemble system

import numpy as np
import forecast_funcs as ff

from pylab import array
from sklearn.externals import joblib

#Load models
pcaConc   = joblib.load('seq_pca_1.pkl')
pcaExtent = joblib.load('seq_pca_2.pkl')
mlpConc   = joblib.load('seq_model_1.pkl')
mlpExtent = joblib.load('seq_model_2.pkl')

#Load the following data:
#   Input: Regional Sea Ice Concentration, thickness, month (June)
#   Ground Truth: Sea Ice Extent for September

numFeat = 18
year    = 2016
month   = 6
fmonth  = 8

iceType = 'extent'
siiVersion ='v3.0'
forecastVar = 'conc'
hemStr = 'N'
rawDataPath = '../../Data/'
derivedDataPath = '../../DataOutput/'
regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')

sample = np.zeros(numFeat)
sample[0] = (month-1)
sample[1] = np.ma.mean(ff.get_pmas_month(rawDataPath, year, month-1))

#set region features to array
iceConc = ff.get_gridvar(derivedDataPath, forecastVar, month, 
                  array(year), hemStr)

for regInd in range(16):
    regionData = iceConc.data
    regionData[iceConc.mask == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    sample[regInd+2] = 100*np.ma.mean(np.multiply(regionData, desiredRegion))
    
sample = np.reshape(sample, (1, -1))
feat = pcaConc.transform(sample)
#feat=sample

_, extent = ff.get_ice_extentN(rawDataPath, fmonth, year, year, iceType, siiVersion, hemStr)
gTruth = extent[0]

#Ensemble Model run

output = mlpConc.predict(feat) #THIS OUTPUT REPRESENTS SEA ICE FOR SEPTEMBER
debug = output

#Testing out penalizing small features as inaccuracies
for i in range(np.size(output)):
    if (output[0][i] < 0.09):
        output[0][i] = 0
        debug[0][i] = 0


month=8
sample = np.zeros(numFeat)
sample[0] = (month-1)
sample[1] = np.ma.mean(ff.get_pmas_month(rawDataPath, year, month-1))

#set region features to array
iceConc = ff.get_gridvar(derivedDataPath, forecastVar, month, 
                  array(year), hemStr)

for regInd in range(16):
    regionData = iceConc.data
    regionData[iceConc.mask == True] = 0 #get rid of nan
    desiredRegion = regionMask == regInd
    sample[regInd+2] = 100*np.ma.mean(np.multiply(regionData, desiredRegion)) 


#for m in range(month+1, fmonth+1): #we want to predict for september
for m in range(7, fmonth+1):    
    
    #Testing out penalizing small features as inaccuracies
    for i in range(np.size(output)):
        if (output[0][i] < 0.09):
            output[0][i] = 0
       
    #output[:, 1:np.size(output)]=output[:, 1:np.size(output)]*100
    m_arr = [m]
    output = np.column_stack((m_arr, output))
    output = pcaConc.transform(output)
    output = mlpConc.predict(output)
 
    

#Predict extent:
output = np.column_stack((fmonth-1, output))
#Testing out penalizing small features as inaccuracies
for i in range(np.size(output)):
    if (output[0][i] < 0.09):
        output[0][i] = 0


#output = pcaExtent.transform(output)
final  = mlpExtent.predict(output)[0]


  
    
    
    